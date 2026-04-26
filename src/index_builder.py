#!/usr/bin/env python3
"""
index_builder.py
PDF -> markdown text -> chunks -> embeddings -> BM25 + FAISS + metadata
"""

import os
import pickle
import pathlib
import re
import json
from typing import List, Dict, Tuple

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from src.embedder import SentenceTransformer

from src.preprocessing.chunking import DocumentChunker, ChunkConfig, print_chunk_stats
from src.preprocessing.extraction import extract_sections_from_markdown

# ----- runtime parallelism knobs (avoid oversubscription) -----
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

DEFAULT_EXCLUSION_KEYWORDS = ['questions', 'exercises', 'summary', 'references']


def build_index(
    markdown_file: str,
    *,
    chunker: DocumentChunker,
    chunk_config: ChunkConfig,
    embedding_model_path: str,
    embedding_model_context_window: int,
    artifacts_dir: os.PathLike,
    index_prefix: str,
    use_multiprocessing: bool = False,
    use_headings: bool = False,
) -> None:
    """
    Extract sections, chunk, embed, and build both FAISS and BM25 indexes.

    Persists:
        - {prefix}.faiss
        - {prefix}_bm25.pkl
        - {prefix}_chunks.pkl
        - {prefix}_sources.pkl
        - {prefix}_meta.pkl
        - {prefix}_page_to_chunk_map.json
    """
    all_chunks: List[str] = []
    sources: List[str] = []
    metadata: List[Dict] = []

    sections = extract_sections_from_markdown(
        markdown_file,
        exclusion_keywords=DEFAULT_EXCLUSION_KEYWORDS
    )

    page_to_chunk_ids = {}
    current_page = 1
    total_chunks = 0
    heading_stack = []

    # Step 1: Chunk
    for i, c in enumerate(sections):
        current_level = c.get('level', 1)
        chapter_num = c.get('chapter', 0)

        while heading_stack and heading_stack[-1][0] >= current_level:
            heading_stack.pop()

        if c['heading'] != "Introduction":
            heading_stack.append((current_level, c['heading']))

        path_list = [h[1] for h in heading_stack]
        full_section_path = " ".join(path_list)
        full_section_path = f"Chapter {chapter_num} " + full_section_path

        sub_chunks = chunker.chunk(c['content'])
        page_pattern = re.compile(r'--- Page (\d+) ---')

        for sub_chunk_id, sub_chunk in enumerate(sub_chunks):
            chunk_pages = set()
            fragments = page_pattern.split(sub_chunk)

            if fragments[0].strip():
                page_to_chunk_ids.setdefault(current_page, set()).add(total_chunks + sub_chunk_id)
                chunk_pages.add(current_page)

            for idx in range(1, len(fragments), 2):
                try:
                    new_page = int(fragments[idx]) + 1
                    if fragments[idx + 1].strip():
                        page_to_chunk_ids.setdefault(new_page, set()).add(total_chunks + sub_chunk_id)
                        chunk_pages.add(new_page)
                    current_page = new_page
                except (IndexError, ValueError):
                    continue

            clean_chunk = re.sub(page_pattern, '', sub_chunk).strip()

            if c["heading"] == "Introduction":
                continue

            meta = {
                "filename": markdown_file,
                "mode": chunk_config.to_string(),
                "char_len": len(clean_chunk),
                "word_len": len(clean_chunk.split()),
                "section": c['heading'],
                "section_path": full_section_path,
                "text_preview": clean_chunk[:100],
                "page_numbers": sorted(list(chunk_pages)),
                "chunk_id": total_chunks + sub_chunk_id,
            }

            chunk_prefix = (
                f"Description: {full_section_path} Content: "
                if use_headings else ""
            )

            all_chunks.append(chunk_prefix + clean_chunk)
            sources.append(markdown_file)
            metadata.append(meta)

        total_chunks += len(sub_chunks)

    # Save page-to-chunk map
    final_map = {page: sorted(list(ids)) for page, ids in page_to_chunk_ids.items()}
    output_file = artifacts_dir / f"{index_prefix}_page_to_chunk_map.json"
    with open(output_file, "w") as f:
        json.dump(final_map, f, indent=2)
    print(f"Saved page to chunk ID map: {output_file}")

    # Print chunk stats before embedding - TODO: wrap in some verbose cfg param
    # print_chunk_stats(all_chunks, chunk_size_in_chars=chunk_config.recursive_chunk_size)

    # Step 2: Load embedder
    print(f"Loading embedding model (n_ctx={embedding_model_context_window})...")
    embedder = SentenceTransformer(
        embedding_model_path,
        n_ctx=embedding_model_context_window,
    )
    print(f"Embedding {len(all_chunks):,} chunks sequentially...")

    if use_multiprocessing:
        print("Starting multi-process pool for embeddings...")
        pool = embedder.start_multi_process_pool(workers=4)
        try:
            embeddings = embedder.encode_multi_process(
                all_chunks,
                pool,
                batch_size=4,
            )
        finally:
            embedder.stop_multi_process_pool(pool)
    else:
        embeddings = embedder.encode(
            all_chunks,
            show_progress_bar=True,
        )

    # Step 3: Build FAISS index
    print(f"Building FAISS index for {len(all_chunks):,} chunks...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, str(artifacts_dir / f"{index_prefix}.faiss"))
    print(f"FAISS index built: {index_prefix}.faiss")

    # Step 4: Build BM25 index
    print(f"Building BM25 index for {len(all_chunks):,} chunks...")
    tokenized_chunks = [preprocess_for_bm25(chunk) for chunk in all_chunks]
    bm25_index = BM25Okapi(tokenized_chunks)
    with open(artifacts_dir / f"{index_prefix}_bm25.pkl", "wb") as f:
        pickle.dump(bm25_index, f)
    print(f"BM25 index built: {index_prefix}_bm25.pkl")

    # Step 5: Persist remaining artifacts
    with open(artifacts_dir / f"{index_prefix}_chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    with open(artifacts_dir / f"{index_prefix}_sources.pkl", "wb") as f:
        pickle.dump(sources, f)
    with open(artifacts_dir / f"{index_prefix}_meta.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print(f"Saved all index artifacts with prefix: {index_prefix}")


def preprocess_for_bm25(text: str) -> list[str]:
    """Lowercase and tokenize text for BM25 indexing."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9_'#+-]", " ", text)

    # Split by whitespace
    tokens = text.split()

    return tokens


# ------------------------ Per-file processing helper -------------------------

def _process_file_to_artifacts(
    markdown_file: str,
    *,
    chunker: DocumentChunker,
    chunk_config: ChunkConfig,
    embedding_model_path: str,
    embedding_model_context_window: int,
    use_multiprocessing: bool = False,
    use_headings: bool = False,
) -> Tuple[List[str], List[str], List[Dict], "np.ndarray", dict]:
    """
    Extract sections, chunk, and embed a single markdown file.
    Returns (all_chunks, sources, metadata, embeddings, page_map).
    Does NOT write any files — callers are responsible for persistence.
    """
    all_chunks: List[str] = []
    sources: List[str] = []
    metadata: List[Dict] = []

    # Extract sections from markdown. Exclude some with certain keywords.
    sections = extract_sections_from_markdown(
        markdown_file,
        exclusion_keywords=DEFAULT_EXCLUSION_KEYWORDS
    )

    page_to_chunk_ids = {}
    current_page = 1
    total_chunks = 0
    heading_stack = []

    # Step 1: Chunk using DocumentChunker
    for i, c in enumerate(sections):
        # Determine current section level
        current_level = c.get('level', 1)

        # Determine current chapter number
        chapter_num = c.get('chapter', 0)

        # Pop sections that are deeper or siblings
        while heading_stack and heading_stack[-1][0] >= current_level:
            heading_stack.pop()

        # Push pair of (level, heading)
        if c['heading'] != "Introduction":
            heading_stack.append((current_level, c['heading']))

        # Construct section path
        path_list = [h[1] for h in heading_stack]
        full_section_path = " ".join(path_list)
        full_section_path = f"Chapter {chapter_num} " + full_section_path

        # Use DocumentChunker to recursively split this section
        sub_chunks = chunker.chunk(c['content'])

        # Regex to find page markers like "--- Page 3 ---"
        page_pattern = re.compile(r'--- Page (\d+) ---')

        # Iterate through each chunk produced from this section
        for sub_chunk_id, sub_chunk in enumerate(sub_chunks):
            # Track all pages this specific chunk touches
            chunk_pages = set()

            # Split the sub_chunk by page markers to see if it
            # spans multiple pages.
            fragments = page_pattern.split(sub_chunk)

            # If there is content before the first page marker,
            # it belongs to the current_page.
            if fragments[0].strip():
                page_to_chunk_ids.setdefault(current_page, set()).add(total_chunks+sub_chunk_id)
                chunk_pages.add(current_page)

            # Process the new pages found within this sub_chunk.
            # Step by 2 where each pair represents (page number, text after it)
            for i in range(1, len(fragments), 2):
                try:
                    # Get the new page number from the marker
                    new_page = int(fragments[i]) + 1

                    # If there is text after this marker, it belongs to the new_page.
                    if fragments[i+1].strip():
                        page_to_chunk_ids.setdefault(new_page, set()).add(total_chunks + sub_chunk_id)
                        chunk_pages.add(new_page)

                    current_page = new_page

                except (IndexError, ValueError):
                    continue

            # Clean sub_chunk by removing page markers
            clean_chunk = re.sub(page_pattern, '', sub_chunk).strip()

            # Skip introduction chunks for embedding
            if c["heading"] == "Introduction":
                continue

            # Prepare metadata
            meta = {
                "filename": markdown_file,
                "mode": chunk_config.to_string(),
                "char_len": len(clean_chunk),
                "word_len": len(clean_chunk.split()),
                "section": c['heading'],
                "section_path": full_section_path,
                "text_preview": clean_chunk[:100],
                "page_numbers": sorted(list(chunk_pages)),
                "chunk_id": total_chunks + sub_chunk_id
            }

            # Prepare chunk with prefix
            if use_headings:
                chunk_prefix = (
                    f"Description: {full_section_path} "
                    f"Content: "
                )
            else:
                chunk_prefix = ""

            all_chunks.append(chunk_prefix+clean_chunk)
            sources.append(markdown_file)
            metadata.append(meta)

        total_chunks += len(sub_chunks)

    # Step 2: Create embeddings for FAISS index
    print(f"Embedding {len(all_chunks):,} chunks with {pathlib.Path(embedding_model_path).stem} (n_ctx={embedding_model_context_window})...")
    embedder = SentenceTransformer(embedding_model_path, n_ctx=embedding_model_context_window)

    if use_multiprocessing:
        print("Starting multi-process pool for embeddings...")
        # Start the pool. Adjust number of workers as needed.
        pool = embedder.start_multi_process_pool(workers=4)
        try:
            # Compute embeddings in parallel
            embeddings = embedder.encode_multi_process(
                all_chunks,
                pool,
                batch_size=32
            )
        finally:
            # Stop the pool to prevent hanging processes
            embedder.stop_multi_process_pool(pool)
    else:
        # Standard single-process embedding
        embeddings = embedder.encode(
            all_chunks,
            batch_size=8,
            show_progress_bar=True,
            convert_to_numpy=True
        )

    # Convert the sets to sorted lists for a clean, predictable output
    page_map = {pg: sorted(list(id_set)) for pg, id_set in page_to_chunk_ids.items()}

    return all_chunks, sources, metadata, embeddings, page_map


# ------------------------ Incremental index builder --------------------------

def build_incremental_index(
    md_files: List[str],
    *,
    chunker: DocumentChunker,
    chunk_config: ChunkConfig,
    embedding_model_path: str,
    embedding_model_context_window: int,
    artifacts_dir: os.PathLike,
    index_prefix: str,
    checkpoint,           # IndexCheckpoint from checkpoint.py
    use_multiprocessing: bool = False,
    use_headings: bool = False,
    verify_artifacts: bool = True,
) -> None:
    """
    Incrementally build (or update) FAISS + BM25 indexes across multiple
    markdown files, skipping any file whose content has not changed since the
    last run (detected via SHA-256 hash stored in a JSON checkpoint).

    Strategy:
    1. Hash every candidate file and compare against checkpoint.
    2. For each new/changed file: embed and save per-file artifacts under
       artifacts_dir/per_file/<artifact_key>_*.pkl
    3. Flush the checkpoint to disk.
    4. Load ALL per-file artifacts (old + new), merge in memory.
    5. Rebuild the combined FAISS, BM25, and metadata artifacts.

    Per-file artifacts stored under per_file/:
        <key>_chunks.pkl
        <key>_sources.pkl
        <key>_meta.pkl
        <key>_embeddings.pkl
    where <key> is the first 12 hex characters of the file's SHA-256 hash.
    """
    from src.checkpoint import hash_file

    artifacts_dir = pathlib.Path(artifacts_dir)
    per_file_dir = artifacts_dir / "per_file"
    per_file_dir.mkdir(exist_ok=True)

    # If the embedding model or context window changed, all cached embeddings are
    # stale — force every file to be reprocessed before comparing hashes.
    if not checkpoint.config_matches(embedding_model_path, embedding_model_context_window):
        print("Embedding config changed — invalidating all cached per-file artifacts.")
        checkpoint._data = {}   # wipe in-memory state; disk is overwritten at step 6
    elif verify_artifacts:
        # Verify previously-built combined artifacts before any pkl is loaded.
        # On mismatch, wipe checkpoint so all files are treated as new.
        try:
            checkpoint.verify_artifacts(artifacts_dir)
        except ValueError as e:
            print(f"WARNING: artifact verification failed: {e}")
            print("Rebuilding full index from scratch...")
            checkpoint._data = {}

    # Step 1: identify new / changed files
    file_hashes = {f: hash_file(f) for f in md_files}
    new_files = [f for f in md_files if checkpoint.needs_processing(f, file_hashes[f])]
    skipped   = [f for f in md_files if not checkpoint.needs_processing(f, file_hashes[f])]

    if skipped:
        print(f"Skipping {len(skipped)} unchanged file(s):")
        for f in skipped:
            print(f"  [cached] {pathlib.Path(f).name}")

    if not new_files:
        print("No new or changed files — index is already up to date.")
        return

    print(f"Processing {len(new_files)} new file(s):")
    for f in new_files:
        print(f"  [new]    {pathlib.Path(f).name}")

    # Step 2: process new files and save per-file artifacts
    for md_file in new_files:
        fhash = file_hashes[md_file]
        artifact_key = fhash[:12]   # short prefix — unique enough per file

        chunks, sources, meta, embeddings, _ = _process_file_to_artifacts(
            md_file,
            chunker=chunker,
            chunk_config=chunk_config,
            embedding_model_path=embedding_model_path,
            embedding_model_context_window=embedding_model_context_window,
            use_multiprocessing=use_multiprocessing,
            use_headings=use_headings,
        )

        with open(per_file_dir / f"{artifact_key}_chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)
        with open(per_file_dir / f"{artifact_key}_sources.pkl", "wb") as f:
            pickle.dump(sources, f)
        with open(per_file_dir / f"{artifact_key}_meta.pkl", "wb") as f:
            pickle.dump(meta, f)
        with open(per_file_dir / f"{artifact_key}_embeddings.pkl", "wb") as f:
            pickle.dump(embeddings, f)

        checkpoint.upsert(md_file, fhash, len(chunks), artifact_key)
        # Save after each file so a crash only re-processes unfinished files next run.
        # _artifacts and _config are not set yet — those are written in step 6.
        checkpoint.save()
        print(f"  Saved per-file artifacts for {pathlib.Path(md_file).name}  (key={artifact_key})")

    # Step 4: load ALL per-file artifacts and merge
    all_chunks:  List[str]  = []
    all_sources: List[str]  = []
    all_meta:    List[Dict] = []
    faiss_index = None

    md_file_set = set(md_files)
    for rec in checkpoint.all_records():
        # Only include files that are still in the current md_files list
        if rec["file_path"] not in md_file_set:
            continue

        key = rec["artifact_key"]
        with open(per_file_dir / f"{key}_chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        with open(per_file_dir / f"{key}_sources.pkl", "rb") as f:
            sources = pickle.load(f)
        with open(per_file_dir / f"{key}_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        with open(per_file_dir / f"{key}_embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)

        # Adjust chunk_ids to be contiguous across all files
        offset = len(all_chunks)
        for local_idx, m in enumerate(meta):
            m["chunk_id"] = offset + local_idx

        all_chunks.extend(chunks)
        all_sources.extend(sources)
        all_meta.extend(meta)

        # Build / grow FAISS index incrementally (no need to vstack)
        if faiss_index is None:
            dim = embeddings.shape[1]
            faiss_index = faiss.IndexFlatL2(dim)
        faiss_index.add(embeddings)

    if faiss_index is None or not all_chunks:
        print("ERROR: no chunks to index after merging per-file artifacts.")
        return

    total = len(all_chunks)
    print(f"Merging complete — {total:,} total chunks across {len(md_files)} file(s).")

    # Step 5: rebuild combined artifacts
    faiss.write_index(faiss_index, str(artifacts_dir / f"{index_prefix}.faiss"))
    print(f"FAISS index saved ({total:,} vectors)")

    # BM25 must always be rebuilt from the full corpus (no incremental support)
    print(f"Rebuilding BM25 index for {total:,} chunks...")
    tokenized_chunks = [preprocess_for_bm25(c) for c in all_chunks]
    bm25_index = BM25Okapi(tokenized_chunks)
    with open(artifacts_dir / f"{index_prefix}_bm25.pkl", "wb") as f:
        pickle.dump(bm25_index, f)
    print(f"BM25 index saved")

    with open(artifacts_dir / f"{index_prefix}_chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    with open(artifacts_dir / f"{index_prefix}_sources.pkl", "wb") as f:
        pickle.dump(all_sources, f)
    with open(artifacts_dir / f"{index_prefix}_meta.pkl", "wb") as f:
        pickle.dump(all_meta, f)

    # Rebuild page_to_chunk_map from updated metadata
    page_map: Dict[int, List[int]] = {}
    for m in all_meta:
        for pg in m.get("page_numbers", []):
            page_map.setdefault(pg, [])
            cid = m["chunk_id"]
            if cid not in page_map[pg]:
                page_map[pg].append(cid)
    for pg in page_map:
        page_map[pg].sort()

    output_file = artifacts_dir / f"{index_prefix}_page_to_chunk_map.json"
    with open(output_file, "w") as f:
        json.dump(page_map, f, indent=2)
    print(f"Saved page to chunk ID map: {output_file}")

    # Step 6: hash all combined artifacts and flush checkpoint — done last so a
    # crash during rebuild leaves the checkpoint untouched (safe to re-run).
    artifact_filenames = [
        f"{index_prefix}.faiss",
        f"{index_prefix}_bm25.pkl",
        f"{index_prefix}_chunks.pkl",
        f"{index_prefix}_sources.pkl",
        f"{index_prefix}_meta.pkl",
        f"{index_prefix}_page_to_chunk_map.json",
    ]
    checkpoint.set_artifact_hashes(artifacts_dir, artifact_filenames)
    checkpoint.set_config(embedding_model_path, embedding_model_context_window)
    checkpoint.save()
    print(f"Checkpoint updated: {checkpoint.checkpoint_path}")

    print(f"Incremental index build complete. Total chunks indexed: {total:,}")
