#!/usr/bin/env python3
"""
Collect every DLCC DBpedia entity URI under v1/dbpedia/ (positives, negatives,
optional positives_hard / negatives_hard lists, train/test splits) and write a deduplicated list in N-Triples
IRI form: one line per entity, e.g. <http://dbpedia.org/resource/Paris>

Default output: seeds/seed_entities.txt (relative to Knowledge-Graphs repo root).

Example:
  uv run scripts/generate_seed_entities.py
  uv run scripts/generate_seed_entities.py --output seeds/my_seeds.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Basenames to scan for anywhere under v1/dbpedia/
ENTITY_LIST_FILES = frozenset(
    {
        "positives.txt",
        "negatives.txt",
        "positives_hard.txt",
        "negatives_hard.txt",
    }
)
LABELED_SPLIT_FILES = frozenset({"train.txt", "test.txt"})


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def to_nt_iri(uri: str) -> str:
    """Return N-Triples style IRI token: <...>."""
    s = uri.strip()
    if not s or s.startswith("#"):
        return ""
    if s.startswith("<") and s.endswith(">"):
        inner = s[1:-1].strip()
        if not inner:
            return ""
        return f"<{inner}>"
    return f"<{s}>"


def parse_labeled_line(line: str) -> str:
    line = line.strip()
    if not line or line.startswith("#"):
        return ""
    if "\t" in line:
        ent, _ = line.split("\t", 1)
    else:
        parts = line.split(None, 1)
        ent = parts[0] if parts else ""
    return ent.strip()


def parse_uri_line(line: str) -> str:
    line = line.strip()
    if not line or line.startswith("#"):
        return ""
    if "\t" in line:
        return line.split("\t", 1)[0].strip()
    return line


def collect_from_file(path: Path, out: set[str]) -> None:
    name = path.name
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            raw = parse_labeled_line(line) if name in LABELED_SPLIT_FILES else parse_uri_line(line)
            nt = to_nt_iri(raw)
            if nt:
                out.add(nt)


def main() -> None:
    root = repo_root()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dbpedia-root",
        type=Path,
        default=root / "v1" / "dbpedia",
        help="Root folder containing tc01..tc12 (default: <repo>/v1/dbpedia)",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=root / "seeds" / "seed_entities.txt",
        help="Output path (default: <repo>/seeds/seed_entities.txt)",
    )
    args = ap.parse_args()
    dbpedia = args.dbpedia_root if args.dbpedia_root.is_absolute() else root / args.dbpedia_root
    out_path = args.output if args.output.is_absolute() else root / args.output

    if not dbpedia.is_dir():
        raise SystemExit(f"Not a directory: {dbpedia}")

    targets = ENTITY_LIST_FILES | LABELED_SPLIT_FILES
    entities: set[str] = set()
    file_count = 0
    for p in dbpedia.rglob("*.txt"):
        if p.name in targets:
            collect_from_file(p, entities)
            file_count += 1

    print(f"Scanned {file_count} files")

    if not entities:
        raise SystemExit(f"No entities found under {dbpedia}; check dataset layout.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(entities)
    out_path.write_text("\n".join(ordered) + "\n", encoding="utf-8")
    print(f"Wrote {len(ordered)} unique entities to {out_path}")


if __name__ == "__main__":
    main()
