# Convert the synthetic ontology dump in N-Triples format to CSV files for nodes and edges, 
# suitable for import into graph databases or visualization tools.
# Use it to import into Graphy. 

import re
from pathlib import Path

import pandas as pd

# rdflib's N-Triples parser rejects IRIs without a ":" (e.g. <I_tc01_761>); many
# synthetic dumps use bare identifiers in angle brackets. Parse those lines directly.
TRIPLE_LINE = re.compile(r"^\s*<([^>]+)>\s+<([^>]+)>\s+<([^>]+)>\s*\.\s*(?:#.*)?$")

ROOT = Path(__file__).resolve().parent
NT_PATH = ROOT / "v1/synthetic_ontology/tc01/synthetic_ontology/ontology.nt"


def iter_nt_triples(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            m = TRIPLE_LINE.match(line)
            if not m:
                raise ValueError(f"Unrecognized N-Triples line: {line[:200]}")
            yield m.group(1), m.group(2), m.group(3)


nodes_set = set()
edges = []

for s_str, p_str, o_str in iter_nt_triples(NT_PATH):
    nodes_set.add(s_str)
    nodes_set.add(o_str)

    edges.append(
        {
            "Source": s_str,
            "Target": o_str,
            "Label": p_str,
            "Type": "Directed",
        }
    )

nodes = [{"Id": node, "Label": node} for node in nodes_set]

nodes_df = pd.DataFrame(nodes)
edges_df = pd.DataFrame(edges)

nodes_df.to_csv("nodes.csv", index=False)
edges_df.to_csv("edges.csv", index=False)

print("Done. Created nodes.csv and edges.csv")
