import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

# Finds likely useful PDF documents outside drive_docs and copies them into
# drive_docs/_auto_discovered so they can be indexed by the app.

DEFAULT_SOURCE_ROOT = Path('.')
DEFAULT_DRIVE_DOCS = Path('./drive_docs')
AUTO_DISCOVERED_SUBDIR = '_auto_discovered'

IGNORE_DIRS = {
    '.git',
    '.venv',
    '.venv-1',
    '__pycache__',
    'node_modules',
    'chroma_db',
}

KEYWORDS = [
    'agttm',
    'qgttm',
    'mutcd',
    'austroads',
    'temporary traffic management',
    'traffic management',
    'traffic control',
    'road safety barrier',
    'worksite',
]


def sha1_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open('rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def score_pdf(path: Path) -> int:
    text = path.name.lower()
    score = 0
    for kw in KEYWORDS:
        if kw in text:
            score += 1
    return score


def should_ignore(path: Path) -> bool:
    parts = {p.lower() for p in path.parts}
    return any(ignored.lower() in parts for ignored in IGNORE_DIRS)


def build_existing_hash_index(drive_docs_root: Path) -> dict[str, str]:
    index = {}
    for p in drive_docs_root.rglob('*.pdf'):
        if should_ignore(p):
            continue
        try:
            digest = sha1_of_file(p)
            index[digest] = str(p)
        except OSError:
            continue
    return index


def discover_and_copy(source_root: Path, drive_docs_root: Path, min_score: int) -> dict:
    auto_dir = drive_docs_root / AUTO_DISCOVERED_SUBDIR
    auto_dir.mkdir(parents=True, exist_ok=True)

    existing_hashes = build_existing_hash_index(drive_docs_root)

    discovered = []
    copied = []
    skipped_duplicates = []

    for pdf in source_root.rglob('*.pdf'):
        if should_ignore(pdf):
            continue

        # Skip files already inside drive_docs.
        try:
            pdf.resolve().relative_to(drive_docs_root.resolve())
            continue
        except ValueError:
            pass

        score = score_pdf(pdf)
        if score < min_score:
            continue

        discovered.append({
            'path': str(pdf),
            'score': score,
        })

        try:
            digest = sha1_of_file(pdf)
        except OSError:
            continue

        if digest in existing_hashes:
            skipped_duplicates.append({
                'path': str(pdf),
                'duplicate_of': existing_hashes[digest],
            })
            continue

        target = auto_dir / pdf.name
        stem = pdf.stem
        suffix = pdf.suffix
        i = 1
        while target.exists():
            target = auto_dir / f"{stem}_{i}{suffix}"
            i += 1

        try:
            shutil.copy2(pdf, target)
        except OSError:
            continue

        existing_hashes[digest] = str(target)
        copied.append({
            'from': str(pdf),
            'to': str(target),
            'score': score,
        })

    return {
        'source_root': str(source_root.resolve()),
        'drive_docs_root': str(drive_docs_root.resolve()),
        'auto_discovered_dir': str(auto_dir.resolve()),
        'min_score': min_score,
        'discovered_count': len(discovered),
        'copied_count': len(copied),
        'duplicate_count': len(skipped_duplicates),
        'copied': copied,
        'duplicates': skipped_duplicates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Discover useful missing PDFs and copy them into drive_docs for indexing.'
    )
    parser.add_argument(
        '--source-root',
        default=str(DEFAULT_SOURCE_ROOT),
        help='Root folder to scan recursively for PDFs (default: current folder).',
    )
    parser.add_argument(
        '--drive-docs',
        default=str(DEFAULT_DRIVE_DOCS),
        help='Drive docs root where indexed PDFs live (default: ./drive_docs).',
    )
    parser.add_argument(
        '--min-score',
        type=int,
        default=1,
        help='Minimum keyword score required for a PDF to be considered useful (default: 1).',
    )
    parser.add_argument(
        '--report',
        default='discover_report.json',
        help='Path to write JSON report (default: discover_report.json).',
    )
    args = parser.parse_args()

    source_root = Path(args.source_root)
    drive_docs = Path(args.drive_docs)

    result = discover_and_copy(source_root, drive_docs, args.min_score)

    report_path = Path(args.report)
    report_path.write_text(json.dumps(result, indent=2), encoding='utf-8')

    print(f"Discovered: {result['discovered_count']}")
    print(f"Copied: {result['copied_count']}")
    print(f"Duplicates skipped: {result['duplicate_count']}")
    print(f"Report: {report_path.resolve()}")


if __name__ == '__main__':
    main()
