import networkx as nx
from gensim.models import Word2Vec
from node2vec.graph import Graph
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


def read_graph():
    if not args.weighted:
        nx_graph = nx.read_edgelist(edgelist_file,  create_using=nx.DiGraph())
        for edge in nx_graph.edges():
            nx_graph[edge[0]][edge[1]]['weight'] = 1
    else:
        nx_graph = nx.read_edgelist(edgelist_file,  data=(('weight', float), ), create_using=nx.DiGraph())

    if not args.directed:
        nx_graph = nx_graph.to_undirected()

    return nx_graph


def learn_node_features(walks, dim, window, workers, epoch, output):
    emb_walks = [[str(n) for n in walk] for walk in walks]
    node_model = Word2Vec(emb_walks, size=dim, window=window, min_count=0, sg=1, workers=workers, iter=epoch)
    node_model.wv.save_word2vec_format(output)


if __name__ == '__main__':
    nx_graph = read_graph()
    graph = Graph(nx_graph, is_directed=nx.is_directed(nx_graph), p=args.p, q=args.q)
    graph.preprocess_transition_probs()
    walks = graph.simulate_walks(num_walks=args.num_walks, walk_length=args.walk_length)
    learn_node_features(walks, args.dim, args.window_size, args.workers, args.epoch, output_file)
