import numpy as np


class Graph:
    def __init__(self, graph, directed, num_walks, walk_length):
        self.graph = graph
        self.directed = directed
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.alias_nodes, self.alias_edges = {}, {}

        for node in graph.nodes():
            neighbors = sorted(graph.neighbors(node))
            probs = [float(graph[node][neighbor]['weight']) for neighbor in neighbors]
            partition = sum(probs)
            probs = [prob / partition for prob in probs]
            self.alias_nodes[node] = self.alias_setup(probs)
        for edge in graph.edges():
            self.alias_edges[(edge[0], edge[1])] = self.alias_edge(edge[1])
            if not directed:
                self.alias_edges[(edge[1], edge[0])] = self.alias_edge(edge[0])

    def walks(self):
        walks = []
        nodes = list(self.graph.nodes())
        for i in range(self.num_walks):
            np.random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(self.walk_length, node))
        return walks

    def node2vec_walk(self, walk_length, start_node):
        single_walk = [start_node]
        neighbors = sorted(self.graph.neighbors(start_node))
        if len(neighbors) > 0:
            next_node = neighbors[self.alias_draw(self.alias_nodes[start_node][0], self.alias_nodes[start_node][1])]
            single_walk.append(next_node)
        while len(neighbors) > 0 and len(single_walk) < walk_length:
            cur = single_walk[-1]
            neighbors = sorted(self.graph.neighbors(cur))
            if len(neighbors) > 0:
                prv = single_walk[-2]
                next_node = neighbors[self.alias_draw(self.alias_edges[(prv, cur)][0], self.alias_edges[(prv, cur)][1])]
                single_walk.append(next_node)
        return single_walk

    def alias_draw(self, j, q):
        k = len(j)
        kk = int(np.floor(np.random.rand() * k))
        if np.random.rand() < q[kk]:
            return kk
        else:
            return j[kk]

    def alias_setup(self, probs):
        j, q = np.zeros(len(probs), dtype=np.int), np.zeros(len(probs), dtype=np.float)
        smaller, larger = [], []
        for kk, prob in enumerate(probs):
            q[kk] = len(probs) * prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()
            j[small] = large
            q[large] -= (1.0 - q[small])
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)
        return j, q

    def alias_edge(self, cur):
        probs = []
        neighbors = sorted(self.graph.neighbors(cur))
        for neighbor in neighbors:
            probs.append(self.graph[cur][neighbor]['weight'])
        partition = sum(probs)
        probs = [float(prob) / partition for prob in probs]
        return self.alias_setup(probs)