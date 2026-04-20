#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_ttl_line(line: str):
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if not line.endswith("."):
        return None
    parts = line[:-1].strip().split(maxsplit=2)
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


def is_uri_token(token: str) -> bool:
    return token.startswith("<") and token.endswith(">")


def load_seed_entities(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="DBpedia object-properties TTL file")
    ap.add_argument("--seeds", type=Path, required=True, help="seed_entities.txt")
    ap.add_argument("--output", type=Path, required=True, help="Output hop1 TTL")
    ap.add_argument("--entities-out", type=Path, required=True, help="Output entity list from hop1")
    args = ap.parse_args()

    seeds = load_seed_entities(args.seeds)
    found_entities: set[str] = set()
    kept = 0
    seen = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.entities_out.parent.mkdir(parents=True, exist_ok=True)

    with args.input.open("r", encoding="utf-8", errors="replace") as fin, \
         args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            seen += 1
            parsed = parse_ttl_line(line)
            if parsed is None:
                continue

            s, p, o = parsed
            if s in seeds or o in seeds:
                fout.write(line)
                kept += 1
                if is_uri_token(s):
                    found_entities.add(s)
                if is_uri_token(o):
                    found_entities.add(o)

    with args.entities_out.open("w", encoding="utf-8") as f:
        for ent in sorted(found_entities):
            f.write(ent + "\n")

    print(f"Seeds loaded: {len(seeds)}")
    print(f"Triples scanned: {seen}")
    print(f"Triples kept: {kept}")
    print(f"Entities in hop1: {len(found_entities)}")
    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.entities_out}")


if __name__ == "__main__":
    main()