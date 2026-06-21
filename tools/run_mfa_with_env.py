from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent
    shim = project_root.parent / "speechocean_pipeline" / "mfa_pythonpath_shim"
    mfa_env = Path(r"D:\Miniconda3\envs\mfa")
    mfa_root = workspace_root / "speechocean_pipeline" / "mfa_work"
    mfa_root.mkdir(parents=True, exist_ok=True)
    os.environ["MFA_ROOT_DIR"] = str(mfa_root)
    numba_cache = mfa_root / "numba_cache"
    numba_cache.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(numba_cache)
    if shim.exists():
        sys.path.insert(0, str(shim))
    os.environ["PATH"] = (
        str(mfa_env / "Library" / "bin")
        + os.pathsep
        + str(mfa_env / "Scripts")
        + os.pathsep
        + str(mfa_env)
        + os.pathsep
        + os.environ.get("PATH", "")
    )
    from montreal_forced_aligner.command_line.mfa import mfa_cli

    sys.argv = ["mfa", *sys.argv[1:]]
    raise SystemExit(mfa_cli())


if __name__ == "__main__":
    main()
