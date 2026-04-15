"""
checkpoint.py
JSON-backed registry that tracks which markdown files have been processed
into index artifacts, keyed by file path and SHA-256 hash.

Checkpoint file format (checkpoint.json)
-----------------------------------------
{
  "<file_path>": {
    "file_hash":    "<sha256 hex>",
    "indexed_at":   "<ISO-8601 UTC>",
    "num_chunks":   <int>,
    "artifact_key": "<first 12 hex chars of hash>"
  },
  ...
}
"""

import hashlib
import json
import pathlib
from datetime import datetime, timezone
from typing import Optional


# --------------------------------------------------------------------------- #
# File hashing                                                                 #
# --------------------------------------------------------------------------- #

def hash_file(path: str) -> str:
    """Return the SHA-256 hex digest of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Checkpoint store                                                             #
# --------------------------------------------------------------------------- #

class IndexCheckpoint:
    """
    JSON-backed checkpoint that records which markdown files have been
    embedded and saved as per-file artifacts.

    On each index run the caller uses needs_processing() to skip files
    whose hash has not changed since the last run.
    """

    def __init__(self, checkpoint_path: pathlib.Path):
        self.checkpoint_path = pathlib.Path(checkpoint_path)
        self._data: dict = self._load()

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def _load(self) -> dict:
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, "r") as f:
                return json.load(f)
        return {}

    def save(self) -> None:
        """Write the in-memory state to disk (call after each upsert batch)."""
        with open(self.checkpoint_path, "w") as f:
            json.dump(self._data, f, indent=2)

    # ------------------------------------------------------------------ #
    # Queries                                                              #
    # ------------------------------------------------------------------ #

    def get_record(self, file_path: str) -> Optional[dict]:
        """Return the stored record for *file_path*, or None if not found."""
        return self._data.get(file_path)

    def needs_processing(self, file_path: str, current_hash: str) -> bool:
        """
        Return True when the file must be (re-)processed:
          - the file has never been indexed before, OR
          - the stored hash differs from *current_hash* (file has changed).
        """
        rec = self.get_record(file_path)
        return rec is None or rec["file_hash"] != current_hash

    def all_records(self) -> list[dict]:
        """Return every record as a list of dicts, ordered by file_path.
        Skips reserved keys (prefixed with _) such as _artifacts."""
        return [
            {"file_path": k, **v}
            for k, v in sorted(self._data.items())
            if not k.startswith("_")
        ]

    def summary(self) -> None:
        """Print a human-readable summary of the checkpoint to stdout."""
        records = self.all_records()
        if not records:
            print("  (checkpoint is empty)")
            return
        for rec in records:
            print(
                f"  {pathlib.Path(rec['file_path']).name:50s}  "
                f"chunks={rec['num_chunks']:5d}  "
                f"hash={rec['file_hash'][:12]}  "
                f"indexed_at={rec['indexed_at']}"
            )

    # ------------------------------------------------------------------ #
    # Mutations                                                            #
    # ------------------------------------------------------------------ #

    def upsert(
        self,
        file_path: str,
        file_hash: str,
        num_chunks: int,
        artifact_key: str,
    ) -> None:
        """Insert or update the record for *file_path* (in memory only; call save() to persist)."""
        self._data[file_path] = {
            "file_hash":    file_hash,
            "indexed_at":   datetime.now(timezone.utc).isoformat(),
            "num_chunks":   num_chunks,
            "artifact_key": artifact_key,
        }

    def set_artifact_hashes(self, artifacts_dir: pathlib.Path, filenames: list) -> None:
        """Compute and store SHA-256 hashes of combined artifact files under _artifacts.
        Only file byte content is hashed — timestamps and metadata are not included,
        so hashes remain valid after zip/unzip."""
        self._data["_artifacts"] = {
            fname: hash_file(str(artifacts_dir / fname))
            for fname in filenames
            if (artifacts_dir / fname).exists()
        }

    def verify_artifacts(self, artifacts_dir: pathlib.Path) -> None:
        """Check each artifact's current hash against the stored _artifacts hashes.
        Raises ValueError listing every mismatch or missing file."""
        stored = self._data.get("_artifacts", {})
        if not stored:
            print("No artifact hashes in checkpoint — skipping verification.")
            return
        mismatches = []
        for fname, expected in stored.items():
            path = artifacts_dir / fname
            if not path.exists():
                mismatches.append(f"  {fname}: file missing")
                continue
            actual = hash_file(str(path))
            if actual != expected:
                mismatches.append(
                    f"  {fname}: expected {expected[:12]}..., got {actual[:12]}..."
                )
        if mismatches:
            raise ValueError("Artifact verification failed:\n" + "\n".join(mismatches))
        print(f"All {len(stored)} artifact(s) verified OK.")
