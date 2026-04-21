# Knowledge Graphs

Experiments for RDF2Vec-style walks and Word2Vec training on N-Triples graphs.

Run Python entrypoints with **`uv run`** from the repository root (see `pyproject.toml`). Console scripts: `rdf2vec-train` → `src.train.train_word2vec:main`, `protograph` → `src.protograph:main`.

---

## Walks

Input graphs are **N-Triples** (`.nt`): lines matching `<subject> <predicate> <object> .` with IRI nodes.

### `src/walk/random_walks.py`

Two **modes** (choose with `--mode`):

| Mode | Description |
|------|-------------|
| `classic` (default) | Random walks on the **entity graph**: each token is a node; edges follow triples (default **undirected**; use `--directed` for subject→object only). |
| `jrdf2vec-duplicate-free` | Matches jRDF2Vec duplicate-free walks: **forward-only** chains, breadth-first expansion per depth, random trimming to `--walks-per-entity`; lines look like `entity p1 o1 p2 o2 ...`. |

**Classic mode — arguments**

```bash
uv run src/walk/random_walks.py <input.nt> <output_walks.txt> \
  --mode classic \
  --num-walks 10000 \
  --walk-length 40 \
  --directed \
  --seed 42
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `--num-walks` | `10000` | Total walks |
| `--walk-length` | `40` | Nodes per walk |
| `--directed` | off | Use directed edges only |
| `--seed` | none | RNG seed |

**jRDF2Vec duplicate-free — arguments**

```bash
uv run src/walk/random_walks.py <input.nt> <output_walks.txt> \
  --mode jrdf2vec-duplicate-free \
  --walks-per-entity 100 \
  --depth 4 \
  --threads 8 \
  --token-format angled \
  --seed 42
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `--walks-per-entity` | `100` | Max walks per entity after each depth trim |
| `--depth` | `4` | Forward hops |
| `--threads` | `max(1, cpu_count//2)` | Parallel workers |
| `--token-format` | `angled` | `angled`: `<s> <p> <o> ...` (matches training/eval in this repo); `bare`: tokens without brackets |
| `--seed` | none | Seeds per-entity trim randomness |

### `src/walk/wl_walks.py`

**1-WL (Weisfeiler–Lehman)**-colored random walks: same undirected/directed entity graph as classic mode, then WL refinement; next-hop can be uniform or biased toward rarer WL colors.

```bash
uv run src/walk/wl_walks.py <input.nt> <output_walks.txt> \
  --num-walks 10000 \
  --walk-length 40 \
  --wl-iterations 5 \
  --until-converged \
  --max-wl-iterations 100 \
  --initial-labels iri \
  --neighbor-policy uniform \
  --tokens nodes \
  --directed \
  --seed 42
```

**Per-entity walks** (instead of `--num-walks`):

```bash
uv run src/walk/wl_walks.py <input.nt> <out.txt> \
  --walks-per-entity 100 \
  --walk-length 10 \
  --enforce-limits
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `--num-walks` | `10000` | Total walks (ignored if `--walks-per-entity` is set) |
| `--walks-per-entity` | none | Exactly *K* walks from each node |
| `--walk-length` | `40` | Nodes per walk (`--enforce-limits` caps length at 20) |
| `--wl-iterations` | `5` | WL rounds (ignored if `--until-converged`) |
| `--until-converged` | off | Run WL until colors stabilize |
| `--max-wl-iterations` | `100` | Cap for convergence mode |
| `--initial-labels` | `iri` | `iri` or `degree` initial colors |
| `--neighbor-policy` | `uniform` | `uniform` or `inverse-wl-freq` |
| `--tokens` | `nodes` | `nodes` (IRIs), `wl` (hash only), `node_wl` (IRI + WL per step) |
| `--directed` | off | Directed edges only |
| `--seed` | none | RNG seed |

### End-to-end walk example

See `scripts/e2e.sh` and `scripts/random_walks.sh` for copy-paste commands.

---

## Training

Training uses **Gensim Word2Vec** in `src/train/train_word2vec.py`. **Single-corpus** mode (`--mode none`) reads one walk file (one space-separated sentence per line) and writes a **PyTorch** checkpoint (`.pt`) with embeddings, `word2idx`, and metadata—compatible with `src/evaluate_embeddings.py`.

### Single corpus (`--mode none`)

```bash
uv run src/train/train_word2vec.py <walks.txt> \
  -o word2vec_gensim.pt \
  --architecture skipgram \
  --dim 128 \
  --window 5 \
  --epochs 5 \
  --negative 5 \
  --min-count 1 \
  --lr 0.025 \
  --min-alpha 0.0001 \
  --workers 0 \
  --seed 42 \
  --loss-every-steps 0
```

Or: `uv run rdf2vec-train <walks.txt> ...` (same module).

| Argument | Default | Meaning |
|----------|---------|---------|
| `walks` | — | Path to walks file (required) |
| `-o`, `--output` | `word2vec_gensim.pt` | Output `.pt` checkpoint |
| `--architecture` (`--arch`) | `skipgram` | `skipgram` or `cbow` |
| `--dim` | `128` | Embedding size (`vector_size`) |
| `--window` | `5` | Context window |
| `--epochs` | `5` | Training epochs |
| `--negative` | `5` | Negative samples (noise contrastive; `hs=0`) |
| `--min-count` | `1` | Drop tokens below this frequency |
| `--lr` | `0.025` | Initial learning rate (`alpha`) |
| `--min-alpha` | `0.0001` | Floor learning rate after decay |
| `--workers` | `0` | Gensim threads (`0` = all cores; use `1` if using step loss) |
| `--seed` | none | Random seed |
| `--loss-every-steps` | `0` | If `> 0`, log loss every N batch jobs (**forces `workers=1`**) |
| `--loss-log`, `--loss-plot` | auto | CSV / PNG for epoch loss (`<stem>_loss.csv/png`) |
| `--loss-steps-log`, `--loss-steps-plot` | auto | Step loss when `--loss-every-steps` `> 0` |
| `--no-plot` | off | Skip loss PNGs |
| `--gensim-log-progress` | off | Verbose Gensim INFO logs |
| `--report-delay` | `1.0` | Seconds between Gensim progress lines (with `--gensim-log-progress`) |

Reference: `scripts/train.sh`, `scripts/e2e.sh` (walks → train → eval).

---

## Pretraining (two-stage RDF2Vec: protograph → instance)

**Pretraining** here means **stage 1**: Word2Vec on **protograph** walks (P1 or P2), then **stage 2**: continue training on **instance graph** walks (optionally with MASCHInE-style initialization from an ontology). Use `--mode p1` or `--mode p2`.

### 1. Build protographs (optional first step)

```bash
uv run protograph \
  --schema path/to/ontology_schema.nt \
  --kg path/to/instance_graph.nt \
  --out-dir output/protographs \
  --p1-name protograph_p1.nt \
  --p2-name protograph_p2.nt
```

| Argument | Meaning |
|----------|---------|
| `--schema` | **Required.** N-Triples with `rdfs:domain`, `rdfs:range`, `rdfs:subClassOf` |
| `--kg` | Optional: filter relations to predicates appearing in this KG |
| `--out-dir` | Where to write P1/P2 `.nt` files |

### 2. Generate walks on P1 or P2

Same as general walks, e.g. duplicate-free on the protograph file:

```bash
uv run src/walk/random_walks.py protograph_p2.nt walks_p2.txt \
  --mode jrdf2vec-duplicate-free \
  --depth 4 \
  --walks-per-entity 100
```

### 3. Instance walks (stage 2 corpus)

Generate duplicate-free walks on the instance graph (same knobs as protograph walks):

```bash
uv run src/walk/random_walks.py v1/synthetic_ontology/tc07/synthetic_ontology/graph.nt walks_instance.txt \
  --mode jrdf2vec-duplicate-free \
  --depth 4 \
  --walks-per-entity 100 \
  --threads 8 \
  --seed 42
```

### 4. Two-stage training (`--mode p1` or `p2`)

```bash
uv run src/train/train_word2vec.py walks_instance.txt \
  --mode p2 \
  --pretrain-walks walks_p2.txt \
  --ontology v1/synthetic_ontology/tc07/synthetic_ontology/ontology.nt \
  --out-dir output/tc07_rdf2vec_proto \
  -o rdf2vec_final.pt \
  --architecture skipgram \
  --dim 128 \
  --window 5 \
  --negative 5 \
  --min-count 1 \
  --lr 0.025 \
  --pretrain-epochs 5 \
  --finetune-epochs 5 \
  --loss-every-steps 0 \
  --maschine-init
```

**Stage 2 walks** are the **positional `walks` argument** (or `--instance-walks`): a text file produced by `random_walks.py` on your instance `.nt`.

| Argument | Default | Meaning |
|----------|---------|---------|
| `walks` | — | Stage-2 instance walks file (`--instance-walks` overrides) |
| `--mode` | — | `p1` or `p2` |
| `--pretrain-walks` | `walks_p1.txt` / `walks_p2.txt` | Stage-1 corpus |
| `--instance-walks` | none | Stage-2 walks (defaults to positional `walks`) |
| `--out-dir` | `.` | Artifacts: `rdf2vec_pretrained.model`, plots, walk outputs |
| `-o`, `--output` | `rdf2vec_final.pt` | Final `.pt` (relative paths resolve under `--out-dir`) |
| `--pretrain-epochs` | `5` | Stage 1 epochs |
| `--finetune-epochs` | `5` | Stage 2 epochs |
| `--skip-pretrain` | off | Load `rdf2vec_pretrained.model` from `--out-dir` instead of training stage 1 |
| `--ontology` | none | For MASCHInE init (`rdf:type`, `rdfs:subClassOf`); else `ontology.nt` beside instance walks if present |
| `--maschine-init` / `--no-maschine-init` | on | Initialize new instance tokens from class vectors when ontology is available |
| `--loss-every-steps`, `--loss-epoch-plot`, `--loss-pretrain-steps-plot`, `--no-loss-plots` | — | Loss logging/plots for two-stage runs |

Example script: `scripts/protograph.sh`.

---

## Evaluation

```bash
uv run src/evaluate_embeddings.py <test.txt> -c <checkpoint.pt>
```

See `scripts/evaluate.sh` and `scripts/e2e.sh` for dataset-specific paths.

---

## DBpedia (RDF2Vec P1 / P2)

Benchmark entity lists live under [`v1/dbpedia/`](v1/dbpedia/). Build **N-Triples** for protographs and walks with the `dbpedia-build` CLI and the shell drivers in [`scripts/dbpedia/`](scripts/dbpedia/README.md):

```bash
uv run dbpedia-build --help
```

Typical flow:

1. **`export-schema`** — extract `rdfs:domain` / `rdfs:range` / `rdfs:subClassOf` from `dbpedia.owl` → `schema_protograph.nt` for `uv run protograph --schema`.
2. **`filter-instance`** — stream a mapping-based dump (`.nt` / `.nt.bz2`) and keep an undirected **k-hop** neighborhood of your entity URI lists → `instance.nt`, then run `random_walks.py` on that file for stage-2 walks.
3. **`filter-types`** — keep `rdf:type` rows for the same seeds → `types_seed.nt`.
4. **`merge-maschine-ontology`** — concatenate schema + types for `train_word2vec.py --ontology` (MASCHInE init).
5. Optional **`fetch-sparql`** — batched `CONSTRUCT` when you do not have local dumps.
6. **`build-v1-instance`** — scan all `v1/dbpedia/**/*.txt` for resource IRIs, then extract a **multi-hop** subgraph from a mapping-based dump into `v1/dbpedia/dbpedia.nt` (see [`scripts/dbpedia/README.md`](scripts/dbpedia/README.md)).

End-to-end examples: [`scripts/dbpedia/run_from_dumps.sh`](scripts/dbpedia/run_from_dumps.sh) (offline dumps) and [`scripts/dbpedia/run_from_sparql.sh`](scripts/dbpedia/run_from_sparql.sh) (public endpoint). Evaluate a checkpoint with [`scripts/dbpedia/eval_checkpoint.sh`](scripts/dbpedia/eval_checkpoint.sh).

---

## Layout

Typical experiment outputs (see also `scripts/e2e.sh`):

```
output/<experiment_name>/<YYYYMMDD_HHMMSS>/
  walks.txt
  rdf2vec_word2vec.pt   # or custom -o name
  run.log
  *_loss.csv / *_loss.png
```

Two-stage runs additionally store `rdf2vec_pretrained.model` (Gensim) under `--out-dir`.
