# DBpedia artifacts for RDF2Vec (P1 / P2)

This folder documents how to build **N-Triples** inputs for the existing pipeline:

1. **Schema (`schema_protograph.nt`)** — `rdfs:domain`, `rdfs:range`, `rdfs:subClassOf` from the DBpedia ontology (for `uv run protograph --schema`).
2. **Instance subgraph (`instance.nt`)** — object triples around benchmark entities (for `random_walks.py` and optional `protograph --kg`).
3. **Types (`types_seed.nt`)** — `rdf:type` rows for seed resources (for MASCHInE init when merged into `--ontology`).
4. **Merged ontology (`ontology_maschine.nt`)** — `merge-maschine-ontology` → pass to `train_word2vec.py --ontology`.

## DLCC paper (Table 2) — example SPARQL

[`dlcc_table2_dbpedia_sparql.sparql`](dlcc_table2_dbpedia_sparql.sparql) copies the **people-domain** illustrative queries from Portisch & Paulheim (DLCC, arXiv:2207.06014). The full multi-domain query set is in the [DL-TC-Generator](https://github.com/janothan/DL-TC-Generator/tree/master/src/main/resources/queries) repository.

## Full `v1/dbpedia` instance graph (`dbpedia.nt`)

All `http://dbpedia.org/resource/...` IRIs appearing in any `v1/dbpedia/**/*.txt` file are collected, then an **undirected k-hop** subgraph is cut from a mapping-based dump (default **two hops** so edges connect seeds through intermediate nodes—an actual graph, not isolated vertices).

```bash
export DBPEDIA_MAPPING=/path/to/mappingbased_objects_wkd_uris_en.nt.bz2
./scripts/dbpedia/build_v1_dbpedia_nt.sh
# or:
uv run dbpedia-build build-v1-instance --mapping-based "$DBPEDIA_MAPPING" --hops 2 -o v1/dbpedia/dbpedia.nt
```

List every unique resource URI (no dump needed):

```bash
uv run dbpedia-build list-v1-entities --v1-root v1/dbpedia -o v1/dbpedia/dbpedia.entities.txt
```

Optional **`--augment-with-sparql`**: after the dump pass, fetch outgoing `?s ?p ?o` (IRI objects) from the DBpedia endpoint for seeds that never appear as **subject** in the extracted file—useful if the dump slice misses rare resources.

**`--sparql-only`** exists for small tests; full `v1` has ~1M+ URIs so the script defaults to requiring a **local dump**.

## CLI (`dbpedia-build`)

```bash
uv run dbpedia-build --help
uv run dbpedia-build export-schema --help
```

## Official dumps (same release for all files)

From the [DBpedia downloads server](https://www.dbpedia.org/resources/downloads/) (pick one version directory, e.g. `2023.09.01`):

- **Ontology:** `ontology/dbpedia.owl` (or `.owl.bz2`) — feed to `export-schema`.
- **Instance graph:** `core-i18n/en/mappingbased_objects_wkd_uris_en.ttl.bz2` (or the `.nt` / `mappingbased_objects` variant for your locale) — feed to `filter-instance`.
- **Instance types:** `core-i18n/en/instance_types_wkd_uris_en.ttl.bz2` (or matching `instance_types` dump) — feed to `filter-types`.

Convert Turtle dumps to `.nt` with [Apache Jena `riot`](https://jena.apache.org/documentation/io/) if you need strict line-based streaming:

```bash
riot --output=nt mappingbased_objects.nt.gz > mappingbased_objects.nt
```

The `dbpedia-build filter-*` commands accept `.nt` or `.nt.bz2` as produced by `bzip2 -dk`.

## Scripts

| Script | Purpose |
|--------|---------|
| [`run_from_dumps.sh`](run_from_dumps.sh) | Full offline path: export schema, filter instance + types, merge, protograph, walks, training. |
| [`run_from_sparql.sh`](run_from_sparql.sh) | No mapping dump: batched `CONSTRUCT` from the public DBpedia SPARQL endpoint (slower, rate limits). |
| [`eval_checkpoint.sh`](eval_checkpoint.sh) | Run `evaluate_embeddings.py` on a `test.txt` split. |

Example (dumps — adjust paths to your DBpedia release):

```bash
export DBPEDIA_ONTOLOGY=/data/dbpedia.owl
export DBPEDIA_MAPPING=/data/mappingbased_objects_wkd_uris_en.nt.bz2
export DBPEDIA_INSTANCE_TYPES=/data/instance_types_wkd_uris_en.nt.bz2
./scripts/dbpedia/run_from_dumps.sh \
  v1/dbpedia/tc07/species/50/positives.txt \
  v1/dbpedia/tc07/species/50/negatives.txt \
  v1/dbpedia/tc07/species/50/train_test/train.txt \
  v1/dbpedia/tc07/species/50/train_test/test.txt
```

Example (SPARQL — still needs a local ontology file unless `SCHEMA_NT` points to a pre-exported `schema_protograph.nt`):

```bash
export DBPEDIA_ONTOLOGY=/data/dbpedia.owl
./scripts/dbpedia/run_from_sparql.sh \
  v1/dbpedia/tc07/species/50/positives.txt \
  v1/dbpedia/tc07/species/50/train_test/train.txt
```

## Entity files

Pass them as **positional arguments** to the shell scripts (or multiple `--entities` paths to `dbpedia-build`). Typical set:

- `positives.txt`, `negatives.txt`, `negatives_hard.txt`
- `train_test/train.txt`, `train_test/test.txt` (tab-separated URI + label)

so that the subgraph and type filter cover everyone you will walk and evaluate.

## Protograph `--kg` filter

Shell scripts pass `--kg` so P1/P2 only contain predicates that appear in your instance subgraph. If you use a **toy schema** that does not share predicate IRIs with DBpedia, the intersection can be empty; set `SKIP_KG=1` to build protographs from the full schema export instead.
