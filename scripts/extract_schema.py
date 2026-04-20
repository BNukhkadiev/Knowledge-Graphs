#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

RDFS_SUBCLASS = "<http://www.w3.org/2000/01/rdf-schema#subClassOf>"
RDFS_DOMAIN = "<http://www.w3.org/2000/01/rdf-schema#domain>"
RDFS_RANGE = "<http://www.w3.org/2000/01/rdf-schema#range>"
RDF_TYPE = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"


def parse_nt_line(line: str):
    line = line.strip()
    if not line or line.startswith("#") or not line.endswith("."):
        return None
    parts = line[:-1].strip().split(maxsplit=2)
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


def load_classes_from_types(types_path: Path) -> set[str]:
    classes: set[str] = set()
    with types_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parsed = parse_nt_line(line)
            if parsed is None:
                continue
            s, p, o = parsed
            if p == RDF_TYPE:
                classes.add(o)
    return classes


def load_properties_from_graph(graph_path: Path) -> set[str]:
    props: set[str] = set()
    with graph_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parsed = parse_nt_line(line)
            if parsed is None:
                continue
            _, p, _ = parsed
            props.add(p)
    return props


def build_subclass_index(ontology_path: Path) -> dict[str, set[str]]:
    parents: dict[str, set[str]] = {}
    with ontology_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parsed = parse_nt_line(line)
            if parsed is None:
                continue
            s, p, o = parsed
            if p == RDFS_SUBCLASS:
                parents.setdefault(s, set()).add(o)
    return parents


def expand_class_ancestors(seed_classes: set[str], parents: dict[str, set[str]], max_depth: int) -> set[str]:
    all_classes = set(seed_classes)
    frontier = set(seed_classes)

    for _ in range(max_depth):
        next_frontier: set[str] = set()
        for cls in frontier:
            for parent in parents.get(cls, set()):
                if parent not in all_classes:
                    all_classes.add(parent)
                    next_frontier.add(parent)
        if not next_frontier:
            break
        frontier = next_frontier

    return all_classes


def extract_schema(
    ontology_path: Path,
    classes: set[str],
    props: set[str],
    output_path: Path,
) -> tuple[int, int, int]:
    kept_total = 0
    kept_subclass = 0
    kept_domain_range = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with ontology_path.open("r", encoding="utf-8", errors="replace") as fin, \
         output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            parsed = parse_nt_line(line)
            if parsed is None:
                continue
            s, p, o = parsed

            keep = False
            if p == RDFS_SUBCLASS and (s in classes or o in classes):
                keep = True
                kept_subclass += 1
            elif p == RDFS_DOMAIN and s in props and o in classes:
                keep = True
                kept_domain_range += 1
            elif p == RDFS_RANGE and s in props and o in classes:
                keep = True
                kept_domain_range += 1

            if keep:
                fout.write(line)
                kept_total += 1

    return kept_total, kept_subclass, kept_domain_range


def write_set(items: set[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in sorted(items):
            f.write(item + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract relevant schema graph from DBpedia ontology")
    ap.add_argument("--ontology", type=Path, required=True, help="Parsed ontology NT file")
    ap.add_argument("--types", type=Path, required=True, help="types_hop2.ttl")
    ap.add_argument("--graph", type=Path, required=True, help="Merged instance graph G.ttl")
    ap.add_argument("--output", type=Path, required=True, help="Output schema.ttl")
    ap.add_argument("--classes-out", type=Path, default=None, help="Optional output for expanded class set")
    ap.add_argument("--props-out", type=Path, default=None, help="Optional output for property set")
    ap.add_argument("--ancestor-depth", type=int, default=3, help="How many superclass levels to include")
    args = ap.parse_args()

    seed_classes = load_classes_from_types(args.types)
    props = load_properties_from_graph(args.graph)

    print(f"Classes from types: {len(seed_classes)}")
    print(f"Properties from graph: {len(props)}")

    parents = build_subclass_index(args.ontology)
    all_classes = expand_class_ancestors(seed_classes, parents, args.ancestor_depth)

    print(f"Expanded classes (with ancestors): {len(all_classes)}")

    kept_total, kept_subclass, kept_domain_range = extract_schema(
        ontology_path=args.ontology,
        classes=all_classes,
        props=props,
        output_path=args.output,
    )

    if args.classes_out is not None:
        write_set(all_classes, args.classes_out)
    if args.props_out is not None:
        write_set(props, args.props_out)

    print(f"Schema triples written: {kept_total}")
    print(f"  subclass triples: {kept_subclass}")
    print(f"  domain/range triples: {kept_domain_range}")
    print(f"Wrote schema to: {args.output}")


if __name__ == "__main__":
    main()