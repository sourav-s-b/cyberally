"""Install the small, versioned dashboard checkpoint bundle on first run."""

from __future__ import annotations

from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = Path(__file__).resolve().parent / "assets" / "dashboard-models.zip"


def install_bundled_models(root: Path = ROOT) -> int:
    """Copy bundled model files into ignored ``blue/results`` paths.

    Existing local files are preserved. Archive paths are checked before any
    files are written so a malformed bundle cannot escape the project root.
    Returns the number of files installed.
    """
    root = Path(root).resolve()
    bundle = root / BUNDLE.relative_to(ROOT)
    if not bundle.is_file():
        return 0

    installed = 0
    with zipfile.ZipFile(bundle) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            destination = (root / member.filename).resolve()
            if root not in destination.parents:
                raise ValueError(f"unsafe path in dashboard model bundle: {member.filename}")
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                with archive.open(member) as source, destination.open("xb") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
            except FileExistsError:
                continue
            installed += 1
    return installed
