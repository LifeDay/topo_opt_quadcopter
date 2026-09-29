"""Run BESO headless in an isolated run directory.

beso_main.py reads beso_conf.py from its own folder (after resolving symlinks), so each
run gets a private copy of the BESO sources plus a generated beso_conf.py. The submodule
in external/beso is never modified.

BESO calls CalculiX through a small wrapper script that sets the thread count and logs
each call's wall time to ccx_times.log, so BESO's own overhead can be measured.
"""

import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .ccx import find_ccx
from .inp_writer import Material

REPO_ROOT = Path(__file__).resolve().parents[2]
BESO_DIR = REPO_ROOT / "external" / "beso"
BESO_SOURCES = ["beso_main.py", "beso_lib.py", "beso_filters.py", "beso_plots.py", "beso_separate.py"]
# Applied to the private copy only: BESO predates NumPy 2, and its displacement plot uses the
# wrong length when a run stops on oscillation (i_plot = i - 1), which crashes at the very end.
SOURCE_PATCHES = [
    ("np.linalg.linalg.norm", "np.linalg.norm"),
    ("plt.plot(range(i + 1), disp_max_cn,", "plt.plot(range(i_plot + 1), disp_max_cn,"),
]
FAST_FILTER = Path(__file__).with_name("beso_fast_filter.py")

CCX_WRAPPER = """#!/bin/bash
export OMP_NUM_THREADS={threads} CCX_NPROC_EQUATION_SOLVER={threads}
start=$(date +%s.%N)
{ccx!r} "$@"
rc=$?
echo "$1 $start $(date +%s.%N) $rc" >> {times!r}
exit $rc
"""


@dataclass
class Domain:
    """One element set in BESO's config. fi: von Mises allowable (MPa) for the solid state."""
    elset: str
    optimized: bool
    fi: float | None = None


def domains_conf(domains: list[Domain], material: Material, void_ratio: float = 1e-6) -> str:
    """beso_conf.py lines for each domain: two states, void (void_ratio × E and density) and solid."""
    out = []
    for d in domains:
        mats = [f"*ELASTIC\\n{material.E * r:g}, {material.nu:g}\\n*DENSITY\\n{material.density * r:g}"
                for r in (void_ratio, 1.0)]
        fi = "[]" if d.fi is None else f'[[("stress_von_Mises", {d.fi / void_ratio:g})], [("stress_von_Mises", {d.fi:g})]]'
        out += [
            f"elset_name = {d.elset!r}",
            f"domain_optimized[elset_name] = {d.optimized}",
            f"domain_density[elset_name] = [{void_ratio:g}, 1.0]",
            "domain_thickness[elset_name] = [1.0, 1.0]",
            "domain_offset[elset_name] = 0.0",
            "domain_orientation[elset_name] = []",
            f"domain_FI[elset_name] = {fi}",
            f'domain_material[elset_name] = ["{mats[0]}", "{mats[1]}"]',
            "domain_same_state[elset_name] = False",
        ]
    return "\n".join(out) + "\n"


@dataclass
class BesoRun:
    run_dir: Path
    log: Path              # BESO's stdout (and ccx's)
    wall_s: float
    ccx_s: list[float]     # wall time of each ccx call, in order

    @property
    def overhead_s(self) -> float:
        return self.wall_s - sum(self.ccx_s)


def newest_vtk(run_dir: Path, settle_s: float = 2.0) -> Path | None:
    """Newest file*.vtk that has not been written to for settle_s (BESO writes it in parts)."""
    now = time.time()
    done = [p for p in Path(run_dir).glob("file*.vtk") if now - p.stat().st_mtime > settle_s]
    return max(done, default=None, key=lambda p: int(re.sub(r"\D", "", p.stem)))


def run_beso(run_dir: Path, inp_file: Path, conf_body: str, log_name: str = "beso_stdout.log",
             threads: int = 8, live_stl: Path | None = None, poll_s: float = 2.0,
             fast_filter: bool = True) -> BesoRun:
    """Copy inp_file into run_dir, write beso_conf.py and run BESO there.

    conf_body is the text of beso_conf.py without `path`, `path_calculix` and `file_name`,
    which are filled in here. threads is ccx's thread count (8 = the physical cores).
    live_stl: if set, the newest iteration's solid is exported there while BESO runs
    (needs "vtk" in save_resulting_format and save_iteration_results > 0).
    fast_filter: replace BESO's "simple" filter with the vectorized one in beso_fast_filter.py.
    """
    run_dir = Path(run_dir).resolve()
    if run_dir.exists():
        shutil.rmtree(run_dir)  # BESO appends to its log and leaves stale iteration files
    beso_copy = run_dir / "_beso"
    beso_copy.mkdir(parents=True)
    for name in BESO_SOURCES:
        source = (BESO_DIR / name).read_text()
        for old, new in SOURCE_PATCHES:
            source = source.replace(old, new)
        if name == "beso_filters.py" and fast_filter:
            source += "\n\n" + FAST_FILTER.read_text()
        (beso_copy / name).write_text(source)

    times_log = run_dir / "ccx_times.log"
    wrapper = beso_copy / "ccx_timed.sh"
    wrapper.write_text(CCX_WRAPPER.format(threads=threads, ccx=find_ccx(), times=str(times_log)))
    wrapper.chmod(0o755)

    shutil.copy2(inp_file, run_dir / Path(inp_file).name)
    header = (
        f"path = {str(run_dir)!r}\n"
        f"path_calculix = {str(wrapper)!r}\n"
        f"file_name = {Path(inp_file).name!r}\n"
    )
    (beso_copy / "beso_conf.py").write_text(header + conf_body)

    env = dict(os.environ, MPLBACKEND="Agg")
    log_path = run_dir / log_name
    exported = None
    start = time.perf_counter()
    with open(log_path, "w") as log:
        proc = subprocess.Popen([sys.executable, str(beso_copy / "beso_main.py")],
                                cwd=run_dir, env=env, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            time.sleep(poll_s)
            if live_stl is not None:
                exported = _export_live(run_dir, live_stl, exported)
    wall = time.perf_counter() - start
    if live_stl is not None:
        _export_live(run_dir, live_stl, exported, settle_s=0.0)
    if proc.returncode != 0:
        raise RuntimeError(f"BESO exited with status {proc.returncode}; see {log_path}")
    ccx_s = [float(end) - float(begin) for _, begin, end, _ in
             (line.split() for line in times_log.read_text().splitlines())]
    return BesoRun(run_dir, log_path, wall, ccx_s)


def _export_live(run_dir: Path, stl: Path, last: Path | None, settle_s: float = 2.0) -> Path | None:
    from .stl_export import beso_solid, export_stl  # pyvista import is slow; only when needed

    vtk = newest_vtk(run_dir, settle_s)
    if vtk is None or vtk == last:
        return last
    try:
        export_stl(beso_solid(vtk), stl)
    except Exception as exc:  # a half-written file; try again next poll
        print(f"live export of {vtk.name} failed: {exc}", file=sys.stderr)
        return last
    return vtk
