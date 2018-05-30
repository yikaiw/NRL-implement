# NRL-implement
Re-implementation of four Network representation learning (NRL) algorithms: DeepWalk, LINE, node2vec, GraphGAN.

## Environment
* NumPy
* TensorFlow
* gensim
* NetworkX

## Data
There are two datasets located in the path ./data/:

* [Cora](https://docs.google.com/spreadsheets/d/1WJ0-2aIhCA37Hj_-Na4umXwYqBGDWKXPeaRj0ECnLw4/edit?usp=sharing): citation dataset.
* [Tencent Weibo](https://docs.google.com/spreadsheets/d/1F1mNarXl8u1CFICm3WufqZrWvCgPkTUqjrXJNnaDsEg/edit#gid=0): following network.


## Training
First, locate at the **root path** of the project:
```
cd NRL-implement
```

For DeepWalk:

```
python DeepWalk/main.py
```
For LINE:

```
python LINE/main.py
```
For node2vec:

```
python node2vec/main.py
```
These three implementations use cora as dataset, and results are saved in ./results/cora/.

Use logistic regression as classifier to evaluate the quality of embeddings produced by these three implementations.

```
python LRclassifier.py --method DeepWalk
```
where DeepWalk can be replaced by LINE and node2vec.


For GraphGAN:

```
python GraphGAN/eval_link_prediction.py
```