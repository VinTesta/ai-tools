from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class ChunkRecord:
    number: int
    title: str
    path: Path
    text: str
    summary: str


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower()).strip("-")
    return slug or "document"


def unique_output_dir(base_dir: Path, source_file: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = f"{slugify(source_file.stem)}-{timestamp}"
    candidate = base_dir / base_name
    counter = 2

    while candidate.exists():
        candidate = base_dir / f"{base_name}-{counter}"
        counter += 1

    return candidate


def split_markdown(markdown: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: langchain-text-splitters. Run `pip install -e .` inside chunk-generator."
        ) from exc

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
        keep_separator=True,
    )
    return splitter.split_text(markdown)


def make_llm(model: str, temperature: float):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY. Export it or run with --no-ai-index.")

    try:
        from langchain_deepseek import ChatDeepSeek
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: langchain-deepseek. Run `pip install -e .` inside chunk-generator."
        ) from exc

    return ChatDeepSeek(model=model, temperature=temperature, api_key=api_key)


def summarize_chunk(llm, source_name: str, chunk_number: int, total_chunks: int, text: str) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(
            content=(
                "Voce gera indices curtos para uma knowledge-base Obsidian. "
                "Resuma o trecho em portugues, preserve nomes, conceitos e secoes importantes. "
                "Maximo 5 bullets curtos. Nao invente informacao."
            )
        ),
        HumanMessage(
            content=(
                f"Documento: {source_name}\n"
                f"Chunk: {chunk_number}/{total_chunks}\n\n"
                f"{text}"
            )
        ),
    ]
    response = llm.invoke(messages)
    return str(response.content).strip()


def fallback_summary(text: str, max_chars: int = 500) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    return f"{compact[:max_chars].rstrip()}..."


def heading_for_chunk(text: str, number: int) -> str:
    for line in text.splitlines():
        clean = line.strip()
        if clean.startswith("#"):
            return clean.lstrip("#").strip() or f"Chunk {number:04d}"
    return f"Chunk {number:04d}"


def write_chunk_file(path: Path, source_file: Path, number: int, total: int, text: str) -> None:
    content = (
        "---\n"
        f"source: {source_file.name}\n"
        f"chunk: {number}\n"
        f"total_chunks: {total}\n"
        "---\n\n"
        f"{text.rstrip()}\n"
    )
    path.write_text(content, encoding="utf-8")


def write_index(path: Path, source_file: Path, chunk_size: int, overlap: int, records: list[ChunkRecord]) -> None:
    lines = [
        f"# Index: {source_file.stem}",
        "",
        f"- Source: `{source_file}`",
        f"- Chunks: {len(records)}",
        f"- Chunk size: {chunk_size}",
        f"- Chunk overlap: {overlap}",
        "",
        "## Chunks",
        "",
    ]

    for record in records:
        rel_path = record.path.as_posix()
        lines.extend(
            [
                f"### {record.number:04d}. {record.title}",
                "",
                f"- File: [{record.path.name}]({rel_path})",
                "- Summary:",
                "",
                record.summary,
                "",
            ]
        )

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="chunk-generator",
        description="Split a Markdown file into chunk files and generate index.md.",
    )
    parser.add_argument("-f", "--file", required=True, help="Input .md file.")
    parser.add_argument("-s", "--chunk-size", required=True, type=int, help="Chunk size in characters.")
    parser.add_argument("-o", "--output-dir", default="generated", help="Base output directory.")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between chunks in characters.")
    parser.add_argument("--model", default="deepseek-chat", help="DeepSeek model name.")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature.")
    parser.add_argument("--no-ai-index", action="store_true", help="Generate index with local text previews only.")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> Path:
    source_file = Path(args.file).expanduser().resolve()
    if not source_file.exists():
        raise ValueError(f"Input file not found: {source_file}")
    if source_file.suffix.lower() != ".md":
        raise ValueError("Input file must be .md")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be greater than 0")
    if args.chunk_overlap < 0:
        raise ValueError("--chunk-overlap must be zero or greater")
    if args.chunk_overlap >= args.chunk_size:
        raise ValueError("--chunk-overlap must be smaller than --chunk-size")
    return source_file


def run(args: argparse.Namespace) -> Path:
    source_file = validate_args(args)
    markdown = source_file.read_text(encoding="utf-8")
    chunks = split_markdown(markdown, args.chunk_size, args.chunk_overlap)
    if not chunks:
        raise ValueError("Input file has no content to chunk.")

    output_base = Path(args.output_dir).expanduser().resolve()
    output_dir = unique_output_dir(output_base, source_file)
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=False)

    llm = None if args.no_ai_index else make_llm(args.model, args.temperature)
    records: list[ChunkRecord] = []
    total = len(chunks)

    for index, chunk in enumerate(chunks, start=1):
        chunk_path = chunks_dir / f"chunk-{index:04d}.md"
        write_chunk_file(chunk_path, source_file, index, total, chunk)
        summary = fallback_summary(chunk) if args.no_ai_index else summarize_chunk(
            llm, source_file.name, index, total, chunk
        )
        records.append(
            ChunkRecord(
                number=index,
                title=heading_for_chunk(chunk, index),
                path=chunk_path.relative_to(output_dir),
                text=chunk,
                summary=summary,
            )
        )

    write_index(output_dir / "index.md", source_file, args.chunk_size, args.chunk_overlap, records)
    return output_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output_dir = run(args)
    except Exception as exc:
        print(f"chunk-generator: error: {exc}", file=sys.stderr)
        return 1

    print(f"Generated: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
