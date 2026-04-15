# Generate protograph 


# Generate walks
uv run src/walk/random_walks.py \
    protograph_p1.nt \
    walks_p1.txt \
    --mode "jrdf2vec-duplicate-free" \
    --depth 4 \
    --walks-per-entity 100

# Instance walks (generate separately; train does not walk the graph)
uv run src/walk/random_walks.py \
    v1/synthetic_ontology/tc07/synthetic_ontology/graph.nt \
    walks_tc07_instance.txt \
    --mode "jrdf2vec-duplicate-free" \
    --depth 4 \
    --walks-per-entity 100

# Train (two-stage RDF2Vec on protograph P1 walks)

uv run src/train/train_word2vec.py \
  --mode p2 \
  --pretrain-walks "walks_p2.txt" \
  --instance-walks "walks.txt" \
  --ontology "v1/synthetic_ontology/tc07/synthetic_ontology/ontology.nt" \
  --out-dir "." \
  --architecture skipgram \
  --dim 100 \
  --window 5 \
  --pretrain-epochs "5" \
  --finetune-epochs "5" \
  --loss-every-steps-pretrain 1 \
  --loss-every-steps-finetune 100 \

# uv run src/train/train_word2vec.py \
#     --mode p2 \
#     --pretrain-walks output/tc07_rdf2vec_proto/walks_p2.txt \
#     --instance-walks walks_tc07_instance.txt \
#     --ontology v1/synthetic_ontology/tc07/synthetic_ontology/ontology.nt \
#     --out-dir output/tc07_rdf2vec_proto
