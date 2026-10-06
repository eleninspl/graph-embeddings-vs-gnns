# Graph Embeddings vs. Graph Neural Networks

A Python framework and empirical study that compares two families of graph-level models on standard graph classification benchmarks: unsupervised graph embeddings (Graph2Vec, DeepWalk, NetLSD) followed by an SVM or MLP classifier, and supervised graph neural networks (GCN, GAT, GIN, PNA). A command-line tool trains any of the models on a dataset in TU Dataset format, extracts embeddings, perturbs graphs to test robustness, and clusters or visualizes the embeddings. The study and its results are written up in a short paper, [*How Powerful Are Classic Graph Neural Networks? A Survey*](paper/paper.pdf).

This is a team project for **Information Systems** (Πληροφοριακά Συστήματα), a 9th-semester course at the School of Electrical and Computer Engineering, National Technical University of Athens (ECE NTUA), academic year 2025–26. It was built by Spiros Maggioros, Reinti Pasai and Eleni Nasopoulou. The original repository is [spirosmaggioros/information_systems](https://github.com/spirosmaggioros/information_systems), and this repository is a fork of it.

According to the commit history, Spiros Maggioros wrote most of the framework (data loading, the GNN models, the training loop and the CLI), and Reinti Pasai wrote NetLSD, the clustering models and the memory measurements. My part was the Graph2Vec and DeepWalk models and their first training code. All three of us are authors of the paper.

| Part of the study | What it measures | Models |
|-------------------|------------------|--------|
| [Classification](#classification) | Accuracy, AUROC and F1 on four datasets | all |
| [Embedding size](#embedding-size) | Accuracy, memory and inference time as the embedding dimension grows | GAT, GIN, PNA, NetLSD, Graph2Vec |
| [Clustering and visualization](#clustering-and-visualization) | How well k-means and spectral clustering recover the classes (ARI), t-SNE plots | all |
| [Robustness](#robustness) | How embeddings and predictions change when edges are added or removed | NetLSD, Graph2Vec, GIN, GAT |

## Getting started

### Prerequisites

- Python 3.10. The code uses `X | Y` type annotations, which need 3.10 or newer, and `karateclub` pins old versions of NumPy and NetworkX that keep the project on 3.10.
- The packages in [`requirements.txt`](requirements.txt): PyTorch, PyTorch Geometric, karateclub, gensim, scikit-learn, Optuna, torchmetrics and matplotlib. Keep SciPy below 1.12, as pinned there. With SciPy 1.13, NetworkX 2.6 cannot build a normalized Laplacian, and every NetLSD trial that uses one fails.
- A GPU is optional. Everything below runs on a CPU. The paper's GNN runs used an NVIDIA A100.

The commands in this README were tested, with fewer epochs, on macOS with Python 3.10.18, PyTorch 2.9.1, PyTorch Geometric 2.7.0, karateclub 1.3.3 and scikit-learn 1.7.2.

```bash
git clone https://github.com/eleninspl/NTUA-InformationSystems.git
cd NTUA-InformationSystems
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the CLI from the repository root with `python -m information_systems`. Use `python -m information_systems <command> --help` to list the options for `train`, `inference` and `analysis`. Model files, JSON results and plots are written to the current folder.

### Datasets

The four datasets are committed under [`data/`](data/). They come from the [TUDataset](https://chrsmrrs.github.io/datasets/) collection, and each folder keeps its original README where one was provided:

| Dataset | Graphs | Classes | Domain | Node attributes |
|---------|--------|---------|--------|-----------------|
| MUTAG | 188 | 2 | Chemical compounds | none |
| ENZYMES | 600 | 6 | Proteins | 18 |
| PROTEINS (full) | 1,113 | 2 | Proteins | 29 |
| IMDB-MULTI | 1,500 | 3 | Actor collaboration networks | none |

The GNNs need node attributes, so they were run only on ENZYMES and PROTEINS. The embedding methods use the graph structure only and were run on all four. The class labels in the committed files were renumbered to start at 0.

## How it works

```
information_systems/__main__.py   CLI: train, inference and analysis commands
dataloader/                       Reads TU Dataset text files into NetworkX graphs, builds PyG batches, perturbs graphs
ml_models/graph_models/           Graph2Vec and DeepWalk (wrapping karateclub), NetLSD, embedding inference
ml_models/torch_geometric/        GCN, GAT (GATv2 layers), GIN and PNA for graph classification
ml_models/classification/         SVM (scikit-learn) and MLP (PyTorch) classifiers for the embeddings
ml_models/clustering/             k-means and spectral clustering
trainer/                          Optuna searches and the PyTorch training loop
visualization/                    t-SNE and Isomap plots, cluster plots, graph drawings
paper/paper.pdf                   The write-up
```

**Embedding methods.** Graph2Vec learns a vector for each graph from its Weisfeiler–Lehman subgraphs, in the same way doc2vec learns a vector for each document from its words. DeepWalk learns node vectors from random walks, and a graph's vector is the mean of its node vectors. NetLSD computes a graph signature from the spectrum of the graph Laplacian, using either the heat or the wave kernel. For these models, `train` runs an Optuna search over the model's hyperparameters: WL iterations and learning rate for Graph2Vec, walk number, walk length and window size for DeepWalk, and kernel, time scales, eigenvalue approximation and normalization for NetLSD. Each trial embeds every graph, splits the embeddings 75/25, trains the chosen classifier and scores it by macro F1. `--epochs` sets the number of trials. The SVM classifier has its own inner Optuna search over kernel and `C`, with 5-fold cross-validation.

**GNNs.** Each GNN stacks message-passing layers, averages the node vectors into a graph vector (mean pooling), and adds a linear layer for classification. The graph vector is the embedding used for clustering and robustness tests. Training uses Adadelta (learning rate 0.1) with early stopping on test F1, and saves the best checkpoint. PNA also needs a histogram of node degrees, which is computed from the training split.

**Metrics.** Training reports accuracy, AUROC and macro F1, and for the GNNs also recall, precision, specificity and the confusion matrix. Inference records the time per graph and the peak memory: CPU memory through `tracemalloc` for the embedding methods, GPU memory for the GNNs.

## Classification

**Task.** Train every model on every dataset it applies to, and compare test accuracy, AUROC and F1, together with inference time and memory.

**Run.** An embedding method with an SVM classifier, on MUTAG:

```bash
python -m information_systems train --model graph2vec --dataset_dir data/MUTAG \
    --classifier SVM --out_channels 128 --epochs 50 \
    --model_name graph2vec_mutag.pkl --classifier_name graph2vec_mutag_svm.pkl
```

The same command works with `--model netlsd` or `--model deepwalk`. A GNN on ENZYMES:

```bash
python -m information_systems train --model gin --dataset_dir data/ENZYMES \
    --num_layers 2 --hidden_channels 64 --out_channels 128 --dropout 0.5 \
    --batch_size 2 --epochs 500 --patience 100 --model_name gin_enzymes.pth
```

The other GNNs are `gcn`, `gat` and `pna`. Add `--device cuda` (or `--device mps` on Apple silicon) to train on a GPU. The paper used 200 Optuna trials for the embedding methods, and 500 epochs with patience 100 and batch size 2 for the GNNs.

**Results.** The paper reports test scores averaged over three training runs. Best accuracy per dataset:

| Dataset | Best GNN | Best embedding method |
|---------|----------|-----------------------|
| MUTAG | – | NetLSD + SVM, 0.92 |
| ENZYMES | GAT, 0.66 | NetLSD + SVM, 0.36 |
| PROTEINS | GIN and PNA, 0.79 | NetLSD + MLP, 0.73 |
| IMDB-MULTI | – | NetLSD + MLP, 0.52 |

On the two datasets with node attributes, the GNNs clearly win. On ENZYMES, the embedding methods reach 0.30 to 0.36 accuracy, against 0.61 to 0.66 for the GNNs. NetLSD with the wave kernel was the strongest embedding method on every dataset. NetLSD took about 4 to 7.5 ms per graph at inference, and Graph2Vec under 1 ms. DeepWalk was much slower (43 ms per graph on MUTAG, 129 ms on PROTEINS), because it trains a new model for every graph.

## Embedding size

**Task.** Measure how accuracy, peak memory and inference time change as the embedding dimension grows.

**Run.** Repeat the training command with different `--hidden_channels` and `--out_channels` (GNNs) or `--out_channels` (embedding methods).

**Results.** Larger embeddings did not reliably help. GAT on ENZYMES stayed between 0.66 and 0.68 accuracy from 32 to 512 output channels, while its peak GPU memory grew from 43 MB to 144 MB. GIN's memory stayed between 48 and 56 MB at every size. NetLSD on MUTAG was best at 64 dimensions (0.91 accuracy). Graph2Vec reached 0.94 at 512 dimensions, but its accuracy did not rise steadily with size (0.92 at 64, 0.89 at 128).

## Clustering and visualization

**Task.** Cluster the embeddings without labels, compare the clusters with the true classes using the adjusted Rand index (ARI), and plot the embeddings in 2D.

**Run.** First save the embeddings of a trained model with `inference`, then analyse them:

```bash
python -m information_systems inference --model graph2vec --out_channels 128 \
    --dataset_dir data/MUTAG --model_weights graph2vec_mutag.pkl \
    --classifier SVM --classifier_weights graph2vec_mutag_svm.pkl \
    --out_json graph2vec_mutag.json

python -m information_systems analysis --in_jsons graph2vec_mutag.json --manifold TSNE
python -m information_systems analysis --in_jsons graph2vec_mutag.json --clustering kmeans
python -m information_systems analysis --in_jsons graph2vec_mutag.json --clustering spectral
```

For a GNN, `inference` must be given the same `--num_layers`, `--hidden_channels` and `--out_channels` that were used for training:

```bash
python -m information_systems inference --model gin --dataset_dir data/ENZYMES \
    --num_layers 2 --hidden_channels 64 --out_channels 128 \
    --model_weights gin_enzymes.pth --out_json gin_enzymes.json
```

`--manifold` accepts `TSNE` or `Isomap` and saves `manifold_visualization.png`. The clustering commands run a short Optuna search over the clustering parameters, print the ARI and show a 2D PCA plot of the clusters.

**Results.** Spectral clustering scored a higher ARI than k-means for 7 of the 9 model and dataset pairs in the paper. ARI stayed low overall. The best score was 0.345 (NetLSD on MUTAG), and on IMDB-MULTI every score was 0.021 or lower, which matches how hard that dataset was to classify.

## Robustness

**Task.** Perturb every graph by adding or removing a percentage of its edges, re-compute the embeddings with an already trained model, and measure how far they move (cosine similarity and Euclidean distance) and how much the accuracy drops.

**Run.** Add `--add_random_edges` or `--remove_random_edges` (a fraction of each graph's edges) to an `inference` command:

```bash
python -m information_systems inference --model gin --dataset_dir data/ENZYMES \
    --num_layers 2 --hidden_channels 64 --out_channels 128 \
    --model_weights gin_enzymes.pth --out_json gin_enzymes_add20.json --add_random_edges 0.2
```

Then compare the `out_features` in the new JSON file with those from the unperturbed run.

**Results.** GIN on PROTEINS lost at most 2 points of accuracy with up to 30% of the edges added or removed. GAT on ENZYMES lost 15 points when 30% of the edges were added. NetLSD with an SVM on MUTAG collapsed: removing 2.5% of the edges dropped accuracy from 0.87 to 0.57, even though the cosine similarity to the original embeddings stayed above 0.97. NetLSD with an MLP on PROTEINS stayed within 2 points. Graph2Vec on PROTEINS degraded gradually, losing up to 23 points.

## Known limitations

The code is kept as the team submitted it. While writing this README, I ran the commands above and read through the code again, and found the following issues:

- **`--shuffle_node_attributes` has no effect.** `shuffle_node_attributes` in `dataloader/dataloader.py` looks up the attribute name `"nøde_attributes"` (with an `ø`), finds nothing, and leaves the graph unchanged. GIN embeddings with and without the flag are identical. The "shuffle attributes" rows in the paper's robustness tables therefore do not measure a shuffle.
- **`--remove_random_edges` is not random.** It removes the first edges in each graph's edge list, so every run removes the same edges. `--add_random_edges` does pick edges at random.
- **Training an embedding method with `--classifier MLP` crashes** at the end, when it saves the classifier, because `save_torch_model` is called with its arguments in the wrong order. The SVM classifier works.
- **Model selection uses the reported split.** The GNN trainer keeps the epoch with the best test F1, and the Optuna search for the embedding methods maximizes the same F1 that it reports, so the reported scores are somewhat optimistic. For the SVM, that F1 comes from a split of the training part, and the 25% test part is not used at all.
- **GNN training ignores `--test_size`** and always holds out 20% of the graphs, not the 25% the paper describes. GNN `inference` also ignores `--device`, so `--device cuda` or `--device mps` fails with a device mismatch.
- **DeepWalk trains a separate model for each graph** and averages its node vectors, so the vectors of different graphs are not in a shared space. DeepWalk's `save` and `load` do nothing, so `inference` re-trains DeepWalk with default hyperparameters instead of the tuned ones.
- **Isolated nodes are dropped.** Graphs are built from their edge lists, so nodes without edges disappear (106 nodes in ENZYMES, 5 in PROTEINS).
- **`pip install .` does not work as described in the original README.** `setup.py` sets `python_requires="<=3.10"`, which pip reads as "3.10.0 or older" and so rejects current 3.10 releases, while the code does not run on 3.9. Its `install_requires` also leaves out several packages the code imports, for example Optuna, torchmetrics and matplotlib. Running `python -m information_systems` from the repository root after `pip install -r requirements.txt` works.
- The `analysis` command accepts several JSON files, but only uses the first one. Comparing a perturbed run with the original is left to the user.

## For students taking the course

This repository is here to help you see how a graph learning study fits together: reading TU datasets, the difference between structure-only embeddings and GNNs that use node features, and how to measure robustness and efficiency. Use it as a reference, not as a template to copy. Build and run your own experiments, and check your own data splits and model selection against the limitations above.

The [PyTorch Geometric documentation](https://pytorch-geometric.readthedocs.io/) and the original papers cited in [`paper/paper.pdf`](paper/paper.pdf) are the best references for each model.
