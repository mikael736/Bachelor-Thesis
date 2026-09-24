"""Snapshot-based run output: each experiment run gets its own directory under
visualisation_output/, containing its plot and an exact copy of the running_code script that
produced it - the snapshot is what makes the run reproducible, no separate metadata needed.
"""
from pathlib import Path


def save_run(run_tag: str, source_code: str, *, output_root: Path) -> Path:
    """Prepare a directory for this run under output_root, writing source_code into it as
    running_code.py. source_code must be read when the script starts, not at save time, since
    the file may have been edited for other runs while this one was computing. Re-running unchanged code reuses the same run_tag directory (a harmless
    overwrite, since the pipeline is deterministic and the output would be identical); a changed
    script sharing the same run_tag gets auto-suffixed (run_tag_2, run_tag_3, ...) instead of
    silently overwriting a different run's snapshot. Returns the directory to save this run's
    outputs into.
    """
    output_root.mkdir(parents=True, exist_ok=True)

    candidate = run_tag
    suffix = 1
    while True:
        run_dir = output_root / candidate
        snapshot_path = run_dir / "running_code.py"
        if not run_dir.exists():
            run_dir.mkdir(parents=True)
            snapshot_path.write_text(source_code, encoding="utf-8")
            return run_dir
        if snapshot_path.exists() and snapshot_path.read_text(encoding="utf-8") == source_code:
            return run_dir
        suffix += 1
        candidate = f"{run_tag}_{suffix}"
