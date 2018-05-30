import numpy as np
import networkx as nx
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import Word2Vec
from graph import Graph
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--embed_dim', type=int, default=128, help=u'Dimension of feature vectors, default 128.')
parser.add_argument('--epoch', type=int, default=20, help=u'Training epochs, default 20.')
parser.add_argument('--workers', type=int, default=4, help=u'Number of parallel workers, default 4.')
parser.add_argument('--walk_length', type=int, default=80, help=u'Length of each walk, default 80.')
parser.add_argument('--num_walks', type=int, default=10, help=u'Number of random walks per node, default 10.')
parser.add_argument('--weighted', type=bool, default=False, help=u'If nodes are weighted, default False.')
parser.add_argument('--directed', type=bool, default=True,
                    help=u'Whether edges are directed, True for cora, default True.')
parser.add_argument('--window_size', type=int, default=10, help=u'Window size, default 10.')
parser.add_argument('--reverse_edges', type=bool, default=True,
                    help=u'Whether edges in edge-list_file need reversing, True for cora, default True.')
args = parser.parse_args()

edgelist_file = '../data/cora/cora.cites'
node_emb_file = '../results/cora/DeepWalk/result.csv'

print('Building networkx graph.', flush=True)
using_graph = nx.DiGraph() if args.directed else nx.Graph()
if not args.weighted:
    nx_graph = nx.read_edgelist(edgelist_file, create_using=using_graph)
    for edge in nx_graph.edges():
        nx_graph[edge[0]][edge[1]]['weight'] = 1
else:
    nx_graph = nx.read_edgelist(edgelist_file, data=(('weight', float),), create_using=using_graph)
if args.directed and args.reverse_edges:
    nx_graph = nx_graph.reverse()

print('Initializing node2vec graph.', flush=True)
graph = Graph(nx_graph, args.directed, args.num_walks, args.walk_length)
print('Processing node2vec walking.', flush=True)
walks = graph.walks()
print('Shape of walks:', np.shape(walks), flush=True)

print('Word2Vec learning.', flush=True)
emb_walks = [[str(w) for w in single_walk] for single_walk in walks]
node_model = Word2Vec(
    emb_walks, size=args.embed_dim, window=args.window_size,
    min_count=0, sg=1, workers=args.workers, iter=args.epoch)
print('Saving results.', flush=True)
node_model.wv.save_word2vec_format(node_emb_file)
