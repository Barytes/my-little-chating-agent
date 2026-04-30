#!/usr/bin/env python3
"""Build a local FAISS index from Markdown notes."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_BASE_URL = "https://space.ai-builders.com/backend/v1"
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_INDEX_PATH = "my_notes.index"
DEFAULT_METADATA_PATH = "my_notes_metadata.json"


@dataclass
class ChunkMetadata:
    source: str
    chunk_index: int
    start_char: int
    end_char: int
    text: str


def find_markdown_files(root: Path) -> list[Path]:
    """Return every Markdown file below root, sorted for stable indexing."""
    return sorted(path for path in root.rglob("*.md") if path.is_file())


def split_text(text: str, chunk_size: int, overlap: int) -> list[tuple[str, int, int]]:
    """Split text into overlapping chunks using paragraph boundaries when possible."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[tuple[str, int, int]] = []
    text_length = len(text)
    start = 0

    while start < text_length:
        hard_end = min(start + chunk_size, text_length)
        end = hard_end

        if hard_end < text_length:
            paragraph_break = text.rfind("\n\n", start, hard_end)
            sentence_break = text.rfind(". ", start, hard_end)
            boundary = max(paragraph_break, sentence_break)
            if boundary > start + chunk_size // 2:
                end = boundary + (2 if boundary == paragraph_break else 1)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append((chunk, start, end))

        if end >= text_length:
            break
        start = max(0, end - overlap)

    return chunks


def load_chunks(markdown_files: Iterable[Path], root: Path, chunk_size: int, overlap: int) -> list[ChunkMetadata]:
    """Read Markdown files and return chunk text plus source metadata."""
    all_chunks: list[ChunkMetadata] = []

    for file_path in markdown_files:
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = file_path.read_text(encoding="utf-8", errors="ignore")

        relative_source = str(file_path.relative_to(root))
        for index, (chunk, start, end) in enumerate(split_text(text, chunk_size, overlap)):
            all_chunks.append(
                ChunkMetadata(
                    source=relative_source,
                    chunk_index=index,
                    start_char=start,
                    end_char=end,
                    text=chunk,
                )
            )

    return all_chunks


def get_api_key() -> str:
    """Read the Student Portal API key from the environment."""
    api_key = os.getenv("BUILDER_API_KEY") or os.getenv("AI_BUILDER_TOKEN")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set BUILDER_API_KEY or AI_BUILDER_TOKEN in your environment or .env file."
        )
    return api_key


def get_openai_client(base_url: str) -> OpenAI:
    """Create an OpenAI-compatible client for the Student Portal backend."""
    return OpenAI(base_url=base_url, api_key=get_api_key())


def embed_chunks(
    client: OpenAI,
    chunks: list[ChunkMetadata],
    model: str,
    batch_size: int,
) -> np.ndarray:
    """Call /embeddings in batches and return float32 embedding vectors."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    vectors: list[list[float]] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        response = client.embeddings.create(
            model=model,
            input=[chunk.text for chunk in batch],
        )
        vectors.extend(item.embedding for item in response.data)
        print(f"Embedded {min(start + batch_size, len(chunks))}/{len(chunks)} chunks")

    return np.array(vectors, dtype="float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a cosine-similarity FAISS index from embedding vectors."""
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty 2D array")

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def save_metadata(chunks: list[ChunkMetadata], metadata_path: Path) -> None:
    """Write chunk metadata next to the FAISS index for later retrieval."""
    metadata_path.write_text(
        json.dumps([asdict(chunk) for chunk in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively index Markdown notes into a local FAISS vector index."
    )
    parser.add_argument("folder", type=Path, help="Folder to recursively scan for .md files")
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_INDEX_PATH), help="FAISS index output path")
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path(DEFAULT_METADATA_PATH),
        help="JSON metadata output path",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model name")
    parser.add_argument("--base-url", default=os.getenv("BUILDER_BASE_URL", DEFAULT_BASE_URL), help="Student Portal API base URL")
    parser.add_argument("--chunk-size", type=int, default=1200, help="Maximum characters per text chunk")
    parser.add_argument("--overlap", type=int, default=200, help="Character overlap between adjacent chunks")
    parser.add_argument("--batch-size", type=int, default=64, help="Number of chunks per embeddings request")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    root = args.folder.expanduser().resolve()

    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Folder does not exist or is not a directory: {root}")

    markdown_files = find_markdown_files(root)
    if not markdown_files:
        raise SystemExit(f"No .md files found under: {root}")

    chunks = load_chunks(markdown_files, root, args.chunk_size, args.overlap)
    if not chunks:
        raise SystemExit(f"Markdown files were found, but no text chunks were created under: {root}")

    print(f"Found {len(markdown_files)} Markdown files")
    print(f"Created {len(chunks)} chunks")

    client = get_openai_client(args.base_url)
    embeddings = embed_chunks(client, chunks, args.model, args.batch_size)
    index = build_faiss_index(embeddings)

    faiss.write_index(index, str(args.output))
    save_metadata(chunks, args.metadata_output)

    print(f"Saved FAISS index to {args.output}")
    print(f"Saved chunk metadata to {args.metadata_output}")


if __name__ == "__main__":
    main()
