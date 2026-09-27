"""Run CalculiX and read results from its .dat file."""

import os
import re
import resource
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HEADER = re.compile(r"^\s*(.+?) \((.+?)\) for set (\S+) and time\s+(\S+)")


@dataclass
class DatResults:
    # (quantity, set) → one array per step/increment, rows = [node id, values...] or [values...]
    blocks: dict[tuple[str, str], list[np.ndarray]] = field(default_factory=dict)
    frequencies_hz: list[float] = field(default_factory=list)

    def get(self, quantity: str, nset: str, step: int = -1) -> np.ndarray:
        return self.blocks[(quantity, nset.upper())][step]


@dataclass
class CcxRun:
    job: Path
    wall_s: float
    max_rss_mb: float
    dat: DatResults


def read_dat(path: Path) -> DatResults:
    res = DatResults()
    lines = Path(path).read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = HEADER.match(line)
        if m:
            key = (m.group(1).strip(), m.group(3).upper())
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            rows = []
            while i < len(lines) and lines[i].strip():
                try:
                    rows.append([float(p) for p in lines[i].split()])
                except ValueError:
                    break
                i += 1
            res.blocks.setdefault(key, []).append(np.array(rows))
            continue
        if "E I G E N V A L U E   O U T P U T" in line:
            i += 1
            while i < len(lines) and "P A R T I C I P" not in lines[i] and "E I G E N" not in lines[i]:
                parts = lines[i].split()
                if len(parts) == 5 and parts[0].isdigit():
                    res.frequencies_hz.append(float(parts[3]))
                i += 1
            continue
        i += 1
    return res


def run_ccx(inp: Path, threads: int | None = None) -> CcxRun:
    """Run ccx on inp (in its folder). Returns wall time, peak memory and parsed .dat.

    Peak memory is RUSAGE_CHILDREN's max RSS: the largest child of this process so far, so
    it is only per-run accurate when runs go from small to large or run in fresh processes.
    """
    inp = Path(inp).resolve()
    ccx = shutil.which("ccx")
    if ccx is None:
        raise FileNotFoundError("CalculiX 'ccx' not found on PATH")
    env = dict(os.environ)
    if threads:
        env["OMP_NUM_THREADS"] = env["CCX_NPROC_EQUATION_SOLVER"] = str(threads)
    start = time.perf_counter()
    with open(inp.with_suffix(".ccx.log"), "w") as log:
        proc = subprocess.run([ccx, "-i", inp.stem], cwd=inp.parent, env=env, stdout=log,
                              stderr=subprocess.STDOUT)
    wall = time.perf_counter() - start
    rss_mb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    log_text = inp.with_suffix(".ccx.log").read_text()
    if proc.returncode != 0 or "*ERROR" in log_text:
        errors = [l for l in log_text.splitlines() if "ERROR" in l][:5]
        raise RuntimeError(f"ccx failed on {inp.name}: {errors or proc.returncode}")
    return CcxRun(inp, wall, rss_mb, read_dat(inp.with_suffix(".dat")))
