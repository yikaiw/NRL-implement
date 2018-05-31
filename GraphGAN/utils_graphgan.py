import numpy as np
from tqdm import tqdm
import copy
import collections


def read_edges(train_filename, test_filename=None):
    train_edges = read_edges_from_file(train_filename)
    test_edges = read_edges_from_file(test_filename) if test_filename else []
    nodes = set()
    linked_nodes = {}
    for edge in train_edges:
        for i in range(2):
            nodes.add(edge[i])
            if edge[i] not in linked_nodes:
                linked_nodes[edge[i]] = []
            linked_nodes[edge[i]].append(edge[1 - i])
    for edge in test_edges:
        for i in range(2):
            nodes.add(edge[i])
            if edge[i] not in linked_nodes:
                linked_nodes[edge[i]] = []
    return len(nodes), linked_nodes


def read_edges_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        edges = [list(map(int, line.split())) for line in lines]
    return edges


def read_pretrained_emd(filename, n_node, n_embed):
    with open(filename, 'r') as f:
        lines = f.readlines()[1:]
    node_embed = np.random.rand(n_node, n_embed)
    for line in lines:
        emd = line.split()
        node_embed[int(float(emd[0])), :] = list(map(float, emd[1:]))
    return node_embed


def construct_tree(nodes, linked_nodes):
    print('Constructing Trees.', flush=True)
    trees = {}
    for root in tqdm(nodes):
        trees[root] = {}
        tmp = copy.copy(linked_nodes[root])
        trees[root][root] = [root] + tmp
        if len(tmp) == 0:  # isolated user
            continue
        queue = collections.deque(tmp)  # the nodes in this queue all are items
        for x in tmp:
            trees[root][x] = [root]
        used_nodes = set(tmp)
        used_nodes.add(root)

        while len(queue) > 0:
            cur_node = queue.pop()
            used_nodes.add(cur_node)
            for sub_node in linked_nodes[cur_node]:
                if sub_node not in used_nodes:
                    queue.appendleft(sub_node)
                    used_nodes.add(sub_node)
                    trees[root][cur_node].append(sub_node)
                    trees[root][sub_node] = [cur_node]
    return trees