#!/usr/bin/env python3
from pathlib import Path
import argparse


def load_entities(path: Path) -> set[str]:
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def parse_line(line: str):
    line = line.strip()
    if not line or not line.endswith("."):
        return None
    parts = line[:-1].strip().split(maxsplit=2)
    if len(parts) != 3:
        return None
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--entities", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    entities = load_entities(args.entities)

    kept = 0
    seen = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.input, "r", encoding="utf-8", errors="replace") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:

        for line in fin:
            seen += 1
            parsed = parse_line(line)
            if parsed is None:
                continue

            s, p, o = parsed

            if s in entities:
                fout.write(line)
                kept += 1

    print(f"Entities: {len(entities)}")
    print(f"Triples scanned: {seen}")
    print(f"Type triples kept: {kept}")


if __name__ == "__main__":
    main()