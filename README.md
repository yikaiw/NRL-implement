# NRL-implement
Re-implement of four Network representation learning (NRL) algorithms: DeepWalk, LINE, node2vec, GraphGAN.

## Python packages requirement
* numpy
* gensim
* networkx

## Data preparing
First, download two datasets:

* [Cora](https://docs.google.com/spreadsheets/d/1WJ0-2aIhCA37Hj_-Na4umXwYqBGDWKXPeaRj0ECnLw4/edit?usp=sharing): citation dataset.
* [Tencent Weibo](https://docs.google.com/spreadsheets/d/1F1mNarXl8u1CFICm3WufqZrWvCgPkTUqjrXJNnaDsEg/edit#gid=0): following network.

Second, build new folder 'data' and put the two datasets into 'data'.


## Training
For node2vec:

```
python -m node2vec
```