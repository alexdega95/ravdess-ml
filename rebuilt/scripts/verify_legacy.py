"""Verify original file contents and symlink destinations against their snapshot."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = Path(__file__).resolve().parent.parent / "docs/legacy_manifest.json"
    entries = json.loads(manifest.read_text())
    differences = []
    for name, expected in entries.items():
        path = args.project_root / name
        if expected["kind"] == "symlink":
            if not path.is_symlink() or os.readlink(path) != expected["target"]:
                differences.append(name)
        elif not path.is_file() or path.is_symlink():
            differences.append(name)
        else:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected["sha256"]:
                differences.append(name)
    if differences:
        print("Original files changed or missing:\n" + "\n".join(differences))
        return 1
    print(f"Verified {len(entries)} original files and links: unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
