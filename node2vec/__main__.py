import networkx as nx
from gensim.models import Word2Vec
from Node2Vec.graph import Graph
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=128, help=u'Dimension of the feature vectors, default 128.')
parser.add_argument('--epoch', type=int, default=20, help=u'Training epochs, default 20.')
parser.add_argument('--workers', type=int, default=4, help=u'Number of parallel workers, default 4.')
parser.add_argument('--walk_length', type=int, default=80, help=u'Length of each walk, default 80.')
parser.add_argument('--num_walks', type=int, default=10, help=u'Number of random walks per node, default 10.')
parser.add_argument('--weighted', type=bool, default=False, help=u'If the nodes are weighted, default False.')
parser.add_argument('--directed', type=bool, default=False, help=u'If the edges are directed, default False.')
parser.add_argument('--window_size', type=int, default=10, help=u'Window size, default 10.')
parser.add_argument('-p', type=float, default=1.0, help=u'Return hyperparameter p, default 1.0.')
parser.add_argument('-q', type=float, default=1.0, help=u'Inout hyperparameter q, default 1.0.')
args = parser.parse_args()

edgelist_file = 'cora.edgelist'
output_file = 'result.emd'

if not args.weighted:
    nx_graph = nx.read_edgelist(edgelist_file,  create_using=nx.DiGraph())
    for edge in nx_graph.edges():
        nx_graph[edge[0]][edge[1]]['weight'] = 1
else:
    nx_graph = nx.read_edgelist(edgelist_file,  data=(('weight', float), ), create_using=nx.DiGraph())
if not args.directed:
    nx_graph = nx_graph.to_undirected()

graph = Graph(nx_graph, args.directed, args.p, args.q, args.num_walks, args.walk_length)
walks = graph.walks()

emb_walks = [[str(w) for w in single_walk] for single_walk in walks]
node_model = Word2Vec(
    emb_walks, size=args.dim, window=args.window_size,
    min_count=0, sg=1, workers=args.workers, iter=args.epoch)
node_model.wv.save_word2vec_format(output_file)
