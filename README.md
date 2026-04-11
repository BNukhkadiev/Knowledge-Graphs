# Knowledge Graphs
Configs store configurations of my experiments


Idea for files:
```
 ├── src                    # 
    ├── train          # 
        ├── w2v_torch.py
        ├── w2v_gensim.py
    ├── eval         # 
    │   ├── evaluate_embeddings.py         # 
    └── walk
       ├── random_walk.py  #
       ├── bfs.py          # 
       └── weisfeiler_lehman.py
```

## Configuration
We can use uv run + .env


## Random Walks
This can be generic because we're curious about what's the comparison vs BFS for example. 
Build a script with:
1. Random Walks
2. Weisfeiler-Lehman algorithm
3. BFS
4. No walks, just the Graph itself.

### Random Walks



Add visuals into latex for each walk on the cases. Actually interesting visualization. 

Random Walk parameters:
1. 
2. 
3. 

----

## Word2Vec
Store as .pt or something.
Enable training from pretrained.
Add another weight initialization. 
Word2Vec params:
1. CBOW vs Skipgram
2. "embedding_dim": 200
3. "window_size": 5,
4. "negative_samples": 10,
5. "learning_rate": 0.001,
5. "epochs": 20,
6. "walk_length": 8,
7. "walks_per_node": 20,
8. "min_count": 2
All of them should be logged.

`.pt` file contains in it already the vocab mapping and embeddings themselves

Gensim word2vec is much faster? Imlpement that one too and compare vs Torch implementation


## ComplEX

## TransE


---

## Evaluation
Training curves are very important. 
Evaluation metrics on the test set. 

Store as an experiment output.
Also write it to the latex.
Also report standard deviation.

add pretty tqdm stats. Make sure they are logged into results file. 


## Results
models will be stored in results/
result folders <dataset>_<walk_type>_<model>_<proto?>/date/:
- embedding weights
- random walks.txt data
- log output
- run config

```
├── output
    ├── <dataset_#>_<walk_type>_<model>/date/
        ├── walks
        ├── config
        ├── model
        ├── metrics
        └── logs
```




## Utils
Create image of a training curve. 



## TODO

It would be nice to do some things:
- 
