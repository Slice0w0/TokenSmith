"""
Unit tests for incremental indexing: hash_file, IndexCheckpoint,
and build_incremental_index.

_process_file_to_artifacts (the embed step) is patched out in all
TestBuildIncrementalIndex tests — it loads a GGUF model and would take minutes.
The .md files still need to physically exist because hash_file() reads them.

Run with: pytest tests/test_checkpoint.py -v
Or:        pytest -m unit
"""

import json
import pickle
import numpy as np
import pytest

from unittest.mock import Mock, patch

# Mark all tests in this module as unit tests
pytestmark = pytest.mark.unit


def _make_fake_artifacts(n_chunks=3, embed_dim=4, filename="test.md"):
    """Mocks for _process_file_to_artifacts
    Return (chunks, sources, metadata, embeddings, page_map) with dummy data."""
    chunks = [f"chunk text {i}" for i in range(n_chunks)]
    sources = [filename] * n_chunks
    metadata = [
        {
            "chunk_id": i,
            "filename": filename,
            "mode": "recursive_sections",
            "char_len": len(chunks[i]),
            "word_len": len(chunks[i].split()),
            "section": "Section A",
            "section_path": "Chapter 1 Section A",
            "text_preview": chunks[i][:100],
            "page_numbers": [i + 1],
        }
        for i in range(n_chunks)
    ]
    embeddings = np.zeros((n_chunks, embed_dim), dtype="float32")
    for i in range(n_chunks):
        embeddings[i, i % embed_dim] = float(i + 1)
    return chunks, sources, metadata, embeddings, {}


def _make_md_files(tmp_path, names):
    """Write dummy markdown files and return their string paths."""
    paths = []
    for name in names:
        p = tmp_path / name
        p.write_text(f"# {name}\nSome dummy content for {name}.")
        paths.append(str(p))
    return paths


def _mock_chunker_and_config():
    mock_chunker = Mock()
    mock_config = Mock()
    mock_config.to_string.return_value = "recursive_sections"
    return mock_chunker, mock_config


# ====================== TestHashFile ======================

class TestHashFile:
    """Tests for the hash_file utility function correctness"""

    def test_hash_is_deterministic(self, tmp_path):
        """hash_file returns the same digest on two successive calls to an unchanged file."""
        from src.checkpoint import hash_file

        f = tmp_path / "file.md"
        f.write_text("hello world")

        assert hash_file(str(f)) == hash_file(str(f))

    def test_different_content_produces_different_hash(self, tmp_path):
        """Two files with different content produce different SHA-256 digests."""
        from src.checkpoint import hash_file

        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        a.write_text("content A")
        b.write_text("content B")

        assert hash_file(str(a)) != hash_file(str(b))

    def test_hash_changes_after_file_modification(self, tmp_path):
        """Modifying file content changes its hash."""
        from src.checkpoint import hash_file

        f = tmp_path / "file.md"
        f.write_text("original content")
        h1 = hash_file(str(f))
        f.write_text("modified content")
        h2 = hash_file(str(f))

        assert h1 != h2


# ====================== TestIndexCheckpoint ======================

class TestIndexCheckpoint:
    """Tests for IndexCheckpoint, 
    including detecting and re-index added/changed file, 
    and skip indexing when existing checkpoint approves it"""

    def test_new_file_needs_processing(self, tmp_path):
        """A file absent from the checkpoint is flagged for processing."""
        from src.checkpoint import IndexCheckpoint

        cp = IndexCheckpoint(tmp_path / "checkpoint.json")

        assert cp.needs_processing("some/file.md", "abc123") is True

    def test_unchanged_file_is_not_reprocessed(self, tmp_path):
        """After upsert, a file with the same hash does not need processing."""
        from src.checkpoint import IndexCheckpoint

        cp = IndexCheckpoint(tmp_path / "checkpoint.json")
        cp.upsert("file.md", "abc123", 10, "abc123abc123")

        assert cp.needs_processing("file.md", "abc123") is False

    def test_changed_file_needs_reprocessing(self, tmp_path):
        """A file whose hash differs from the stored record is flagged for reprocessing."""
        from src.checkpoint import IndexCheckpoint

        cp = IndexCheckpoint(tmp_path / "checkpoint.json")
        cp.upsert("file.md", "oldhash", 10, "oldhash00000")

        assert cp.needs_processing("file.md", "newhash") is True

    def test_save_and_reload_persists_records(self, tmp_path):
        """Records survive a save() followed by loading a fresh IndexCheckpoint."""
        from src.checkpoint import IndexCheckpoint

        path = tmp_path / "checkpoint.json"
        cp = IndexCheckpoint(path)
        cp.upsert("file.md", "abc123", 5, "abc123abc123")
        cp.save()

        # Load from the same file in a fresh instance
        cp2 = IndexCheckpoint(path)
        rec = cp2.get_record("file.md")

        assert rec is not None
        assert rec["file_hash"] == "abc123"
        assert rec["num_chunks"] == 5
        assert rec["artifact_key"] == "abc123abc123"

    def test_upsert_overwrites_existing_record(self, tmp_path):
        """A second upsert for the same path replaces the previous entry."""
        from src.checkpoint import IndexCheckpoint

        cp = IndexCheckpoint(tmp_path / "checkpoint.json")
        cp.upsert("file.md", "hash_v1", 10, "hash_v1000000")
        cp.upsert("file.md", "hash_v2", 20, "hash_v2000000")
        rec = cp.get_record("file.md")

        assert rec["file_hash"] == "hash_v2"
        assert rec["num_chunks"] == 20

    def test_persisted_file_is_valid_json(self, tmp_path):
        """The checkpoint file written by save() is valid, parseable JSON."""
        from src.checkpoint import IndexCheckpoint

        path = tmp_path / "checkpoint.json"
        cp = IndexCheckpoint(path)
        cp.upsert("file.md", "abc", 3, "abcabcabcabc")
        cp.save()

        with open(path) as f:
            data = json.load(f)

        assert "file.md" in data


# ====================== TestBuildIncrementalIndex ======================

class TestBuildIncrementalIndex:
    """Tests for build_incremental_index with _process_file_to_artifacts patched out."""

    @patch('src.index_builder._process_file_to_artifacts')
    def test_new_file_is_processed(self, mock_process, tmp_path):
        """A file absent from the checkpoint is passed to _process_file_to_artifacts."""
        from src.checkpoint import IndexCheckpoint
        from src.index_builder import build_incremental_index

        md_files = _make_md_files(tmp_path, ["chap1.md"])
        mock_process.return_value = _make_fake_artifacts(filename=md_files[0])

        cp = IndexCheckpoint(tmp_path / "checkpoint.json")
        chunker, chunk_config = _mock_chunker_and_config()
        build_incremental_index(
            md_files=md_files,
            chunker=chunker, chunk_config=chunk_config,
            embedding_model_path="dummy/model.gguf",
            artifacts_dir=tmp_path, index_prefix="test_index",
            checkpoint=cp,
        )

        assert mock_process.call_count == 1
        assert mock_process.call_args[0][0] == md_files[0]

    @patch('src.index_builder._process_file_to_artifacts')
    def test_unchanged_file_skipped_on_second_run(self, mock_process, tmp_path):
        """A file already in the checkpoint with the same content hash is not re-embedded."""
        from src.checkpoint import IndexCheckpoint
        from src.index_builder import build_incremental_index

        md_files = _make_md_files(tmp_path, ["chap1.md"])
        mock_process.return_value = _make_fake_artifacts(filename=md_files[0])
        chunker, chunk_config = _mock_chunker_and_config()

        # First run — processes the file
        cp = IndexCheckpoint(tmp_path / "checkpoint.json")
        build_incremental_index(
            md_files=md_files,
            chunker=chunker, chunk_config=chunk_config,
            embedding_model_path="dummy/model.gguf",
            artifacts_dir=tmp_path, index_prefix="test_index",
            checkpoint=cp,
        )
        assert mock_process.call_count == 1

        # Second run — same file, same content on disk
        cp2 = IndexCheckpoint(tmp_path / "checkpoint.json")
        build_incremental_index(
            md_files=md_files,
            chunker=chunker, chunk_config=chunk_config,
            embedding_model_path="dummy/model.gguf",
            artifacts_dir=tmp_path, index_prefix="test_index",
            checkpoint=cp2,
        )

        # No additional embedding call should have been made
        assert mock_process.call_count == 1

    @patch('src.index_builder._process_file_to_artifacts')
    def test_only_new_file_processed_when_second_file_added(self, mock_process, tmp_path):
        """When a second file is added, only that file is passed to _process_file_to_artifacts."""
        from src.checkpoint import IndexCheckpoint
        from src.index_builder import build_incremental_index

        file_a = _make_md_files(tmp_path, ["chap1.md"])[0]
        chunker, chunk_config = _mock_chunker_and_config()

        # First run: index file_a
        mock_process.return_value = _make_fake_artifacts(n_chunks=3, filename=file_a)
        cp = IndexCheckpoint(tmp_path / "checkpoint.json")
        build_incremental_index(
            md_files=[file_a],
            chunker=chunker, chunk_config=chunk_config,
            embedding_model_path="dummy/model.gguf",
            artifacts_dir=tmp_path, index_prefix="test_index",
            checkpoint=cp,
        )
        assert mock_process.call_count == 1

        # Second run: add file_b — only file_b should be re-embedded
        file_b = _make_md_files(tmp_path, ["chap2.md"])[0]
        mock_process.return_value = _make_fake_artifacts(n_chunks=2, filename=file_b)
        cp2 = IndexCheckpoint(tmp_path / "checkpoint.json")
        build_incremental_index(
            md_files=[file_a, file_b],
            chunker=chunker, chunk_config=chunk_config,
            embedding_model_path="dummy/model.gguf",
            artifacts_dir=tmp_path, index_prefix="test_index",
            checkpoint=cp2,
        )

        assert mock_process.call_count == 2
        assert mock_process.call_args[0][0] == file_b

    @patch('src.index_builder._process_file_to_artifacts')
    def test_chunk_ids_are_contiguous_across_files(self, mock_process, tmp_path):
        """chunk_id values in merged metadata run 0..N-1 with no gaps or duplicates."""
        from src.checkpoint import IndexCheckpoint
        from src.index_builder import build_incremental_index

        file_a, file_b = _make_md_files(tmp_path, ["chap1.md", "chap2.md"])
        chunker, chunk_config = _mock_chunker_and_config()

        # Both files are new — process them in a single run
        mock_process.side_effect = [
            _make_fake_artifacts(n_chunks=3, filename=file_a),
            _make_fake_artifacts(n_chunks=2, filename=file_b),
        ]
        cp = IndexCheckpoint(tmp_path / "checkpoint.json")
        build_incremental_index(
            md_files=[file_a, file_b],
            chunker=chunker, chunk_config=chunk_config,
            embedding_model_path="dummy/model.gguf",
            artifacts_dir=tmp_path, index_prefix="test_index",
            checkpoint=cp,
        )

        with open(tmp_path / "test_index_meta.pkl", "rb") as f:
            combined_meta = pickle.load(f)

        ids = [m["chunk_id"] for m in combined_meta]
        assert ids == list(range(len(combined_meta)))

    @patch('src.index_builder._process_file_to_artifacts')
    def test_combined_chunk_count_matches_sum_of_files(self, mock_process, tmp_path):
        """Combined chunks pkl contains exactly the sum of all per-file chunk counts."""
        from src.checkpoint import IndexCheckpoint
        from src.index_builder import build_incremental_index

        file_a, file_b = _make_md_files(tmp_path, ["chap1.md", "chap2.md"])
        chunker, chunk_config = _mock_chunker_and_config()

        mock_process.side_effect = [
            _make_fake_artifacts(n_chunks=4, filename=file_a),
            _make_fake_artifacts(n_chunks=5, filename=file_b),
        ]
        cp = IndexCheckpoint(tmp_path / "checkpoint.json")
        build_incremental_index(
            md_files=[file_a, file_b],
            chunker=chunker, chunk_config=chunk_config,
            embedding_model_path="dummy/model.gguf",
            artifacts_dir=tmp_path, index_prefix="test_index",
            checkpoint=cp,
        )

        with open(tmp_path / "test_index_chunks.pkl", "rb") as f:
            combined_chunks = pickle.load(f)

        assert len(combined_chunks) == 4 + 5

    @patch('src.index_builder._process_file_to_artifacts')
    def test_all_expected_artifact_files_are_created(self, mock_process, tmp_path):
        """All required combined artifact files exist after a successful index build."""
        from src.checkpoint import IndexCheckpoint
        from src.index_builder import build_incremental_index

        md_files = _make_md_files(tmp_path, ["chap1.md"])
        mock_process.return_value = _make_fake_artifacts(filename=md_files[0])
        chunker, chunk_config = _mock_chunker_and_config()

        cp = IndexCheckpoint(tmp_path / "checkpoint.json")
        build_incremental_index(
            md_files=md_files,
            chunker=chunker, chunk_config=chunk_config,
            embedding_model_path="dummy/model.gguf",
            artifacts_dir=tmp_path, index_prefix="test_index",
            checkpoint=cp,
        )

        expected = [
            "test_index.faiss",
            "test_index_bm25.pkl",
            "test_index_chunks.pkl",
            "test_index_sources.pkl",
            "test_index_meta.pkl",
            "test_index_page_to_chunk_map.json",
            "checkpoint.json",
        ]
        for fname in expected:
            assert (tmp_path / fname).exists(), f"Missing artifact: {fname}"

    @patch('src.index_builder._process_file_to_artifacts')
    def test_no_changes_does_not_call_process_again(self, mock_process, tmp_path):
        """When all files are cached, _process_file_to_artifacts is never called again."""
        from src.checkpoint import IndexCheckpoint
        from src.index_builder import build_incremental_index

        md_files = _make_md_files(tmp_path, ["chap1.md"])
        mock_process.return_value = _make_fake_artifacts(filename=md_files[0])
        chunker, chunk_config = _mock_chunker_and_config()

        # First run — builds index
        cp = IndexCheckpoint(tmp_path / "checkpoint.json")
        build_incremental_index(
            md_files=md_files,
            chunker=chunker, chunk_config=chunk_config,
            embedding_model_path="dummy/model.gguf",
            artifacts_dir=tmp_path, index_prefix="test_index",
            checkpoint=cp,
        )

        # Second run — nothing changed on disk
        cp2 = IndexCheckpoint(tmp_path / "checkpoint.json")
        build_incremental_index(
            md_files=md_files,
            chunker=chunker, chunk_config=chunk_config,
            embedding_model_path="dummy/model.gguf",
            artifacts_dir=tmp_path, index_prefix="test_index",
            checkpoint=cp2,
        )

        assert mock_process.call_count == 1

    @patch('src.index_builder._process_file_to_artifacts')
    def test_checkpoint_records_all_processed_files(self, mock_process, tmp_path):
        """After indexing, every processed file has a record in the checkpoint."""
        from src.checkpoint import IndexCheckpoint
        from src.index_builder import build_incremental_index

        file_a, file_b = _make_md_files(tmp_path, ["chap1.md", "chap2.md"])
        chunker, chunk_config = _mock_chunker_and_config()

        mock_process.side_effect = [
            _make_fake_artifacts(n_chunks=2, filename=file_a),
            _make_fake_artifacts(n_chunks=2, filename=file_b),
        ]
        cp = IndexCheckpoint(tmp_path / "checkpoint.json")
        build_incremental_index(
            md_files=[file_a, file_b],
            chunker=chunker, chunk_config=chunk_config,
            embedding_model_path="dummy/model.gguf",
            artifacts_dir=tmp_path, index_prefix="test_index",
            checkpoint=cp,
        )

        # Reload and verify both files are recorded
        cp2 = IndexCheckpoint(tmp_path / "checkpoint.json")
        assert cp2.get_record(file_a) is not None
        assert cp2.get_record(file_b) is not None
