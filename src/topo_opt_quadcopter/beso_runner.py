"""Run BESO headless in an isolated run directory.

beso_main.py reads beso_conf.py from its own folder (after resolving symlinks), so each
run gets a private copy of the BESO sources plus a generated beso_conf.py. The submodule
in external/beso is never modified.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .ccx import find_ccx

REPO_ROOT = Path(__file__).resolve().parents[2]
BESO_DIR = REPO_ROOT / "external" / "beso"
BESO_SOURCES = ["beso_main.py", "beso_lib.py", "beso_filters.py", "beso_plots.py", "beso_separate.py"]
# BESO predates NumPy 2; these are applied to the private copy only.
NUMPY2_PATCHES = [("np.linalg.linalg.norm", "np.linalg.norm")]


def run_beso(run_dir: Path, inp_file: Path, conf_body: str, log_name: str = "beso_stdout.log") -> Path:
    """Copy inp_file into run_dir, write beso_conf.py and run BESO there.

    conf_body is the text of beso_conf.py without `path`, `path_calculix` and `file_name`,
    which are filled in here. Returns the path of the stdout log.
    """
    run_dir = Path(run_dir).resolve()
    if run_dir.exists():
        shutil.rmtree(run_dir)  # BESO appends to its log and leaves stale iteration files
    beso_copy = run_dir / "_beso"
    beso_copy.mkdir(parents=True)
    for name in BESO_SOURCES:
        source = (BESO_DIR / name).read_text()
        for old, new in NUMPY2_PATCHES:
            source = source.replace(old, new)
        (beso_copy / name).write_text(source)

    shutil.copy2(inp_file, run_dir / Path(inp_file).name)
    header = (
        f"path = {str(run_dir)!r}\n"
        f"path_calculix = {find_ccx()!r}\n"
        f"file_name = {Path(inp_file).name!r}\n"
    )
    (beso_copy / "beso_conf.py").write_text(header + conf_body)

    env = dict(os.environ, MPLBACKEND="Agg")
    log_path = run_dir / log_name
    with open(log_path, "w") as log:
        result = subprocess.run(
            [sys.executable, str(beso_copy / "beso_main.py")],
            cwd=run_dir, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        raise RuntimeError(f"BESO exited with status {result.returncode}; see {log_path}")
    return log_path
