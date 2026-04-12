uv run src/walk/random_walks.py \
    v1/synthetic_ontology/tc07/synthetic_ontology/graph.nt \
    walks.txt \
    --mode "jrdf2vec-duplicate-free" \
    --depth 4 \
    --walks-per-entity 100
