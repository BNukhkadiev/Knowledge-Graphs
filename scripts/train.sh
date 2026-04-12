uv run src/train/train_word2vec.py \
    walks.txt \
    --architecture skipgram \
    --dim 100 \
    --window 5 \
    --epochs 5 \
