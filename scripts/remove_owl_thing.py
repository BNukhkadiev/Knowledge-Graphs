#!/usr/bin/env python3
from pathlib import Path
import argparse

OWL_THING = "<http://www.w3.org/2002/07/owl#Thing>"


def main():
    ap = argparse.ArgumentParser(description="Remove triples involving owl:Thing")
    ap.add_argument("--input", type=Path, required=True, help="Input schema file")
    ap.add_argument("--output", type=Path, required=True, help="Output cleaned schema file")
    args = ap.parse_args()

    removed = 0
    kept = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.input.open("r", encoding="utf-8", errors="replace") as fin, \
         args.output.open("w", encoding="utf-8") as fout:

        for line in fin:
            if OWL_THING in line:
                removed += 1
                continue

            fout.write(line)
            kept += 1

    print(f"Removed triples: {removed}")
    print(f"Kept triples: {kept}")
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()