import numpy as np
import tensorflow as tf
from tqdm import tqdm
from discriminator import Dis
from generator import Gen
import utils_graphgan as utils
import config as config
import eval_link_prediction as elp

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


class GraphGan(object):
    def __init__(self):
        self.node_num, self.linked_nodes = utils.read_edges(config.train_filename, config.test_filename)
        self.root_nodes = np.arange(self.node_num)
        self.build_gan()
        self.trees = utils.construct_tree(self.root_nodes, self.linked_nodes)
        self.tf_config()

    def build_gan(self):
        print('Building the gan network', flush=True)
        with tf.variable_scope('generator'):
            self.gen = Gen(self.node_num, config.embed_dim, config.lambda_gen, config.lr_gen)
        with tf.variable_scope('discriminator'):
            self.dis = Dis(self.node_num, config.embed_dim, config.lambda_dis, config.lr_dis)

    def tf_config(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess = tf.Session(config=config)
        self.sess.run(init_op)
        self.saver = tf.train.Saver()

    def sample_for_gan(self, root, tree, sample_num, all_score, sample_for_dis):
        assert sample_for_dis in [False, True]
        sample = []
        self.trace = []
        n = 0
        while len(sample) < sample_num:
            node_select = root
            node_father = -1
            self.trace.append([])
            flag = 1
            self.trace[n].append(node_select)
            while True:
                if flag == 1:
                    node_neighbor = tree[node_select][1:]
                else:
                    node_neighbor = tree[node_select]
                flag = 0
                if node_neighbor == []:  # the tree only has the root
                    return sample
                if  sample_for_dis == True:
                    if node_neighbor == [root]:
                        return sample
                    if root in node_neighbor:
                        node_neighbor.remove(root)
                if len(node_neighbor) == 0:
                    return sample
                prob = softmax(all_score[node_select, node_neighbor])
                if np.sum(prob) - 1 > 0.001:
                    print(prob)
                node_check = np.random.choice(node_neighbor, size=1, p=prob)[0]
                self.trace[n].append(node_check)
                if node_check == node_father:
                    sample.append(node_select)
                    break
                node_father = node_select
                node_select = node_check
            n = n + 1
        return sample

    def generate_for_d(self):
        self.samples_rel = []
        self.samples_q = []
        self.samples_label = []
        all_score = self.sess.run(self.gen.all_score)
        for u in self.root_nodes:
            if np.random.rand() < config.update_ratio:  #
                pos = self.linked_nodes[u]  # pos samples
                if len(pos) < 1:
                    continue
                self.samples_rel.extend(pos)
                self.samples_label.extend(len(pos) * [1])
                self.samples_q.extend(len(pos) * [u])
                neg = self.sample_for_gan(u, self.trees[u], len(pos), all_score, sample_for_dis=True)
                if len(neg) < len(pos):
                    continue
                self.samples_rel.extend(neg)
                self.samples_label.extend(len(neg)*[0])
                self.samples_q.extend(len(pos) * [u])

    def get_batch_data(self, index, size):
        q_node = self.samples_q[index:index+size]
        rel_node = self.samples_rel[index:index+size]
        label = self.samples_label[index:index+size]

        return q_node, rel_node, label

    def train(self):
        ckpt = tf.train.get_checkpoint_state(config.model_log)
        if ckpt and ckpt.model_checkpoint_path and config.load_model:
            print('Load the checkpoint: %s.' % ckpt.model_checkpoint_path, flush=True)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        print('Evaluation.', flush=True)
        self.write_emb_to_txt()
        self.eval_test()  # evaluation
        for epoch in tqdm(range(config.max_epochs)):
            #  save the model
            if epoch % config.save_steps == 0 and epoch > 0:
                self.saver.save(self.sess, config.model_log + 'model.ckpt')

            for d_epoch in tqdm(range(config.max_epochs_dis)):
                if d_epoch % config.gen_for_d_iters == 0:  # every gen_for_d_iters round, we generate new data
                    self.generate_for_d()
                train_size = len(self.samples_q)
                #  traverse the whole training dataset sequentially, train the discriminator
                index = 0
                while True:
                    if index > train_size:
                        break
                    if index + config.batch_size_dis <= train_size + 1:
                        input_q_node, input_rel_node, input_label = self.get_batch_data(index, config.batch_size_dis)
                    else:
                        input_q_node, input_rel_node, input_label = self.get_batch_data(index, train_size - index)
                    index += config.batch_size_dis
                    _, loss = self.sess.run([self.dis.d_updates, self.dis.pre_loss], {self.dis.q_node: np.array(input_q_node),\
                                      self.dis.rel_node: np.array(input_rel_node), self.dis.label: np.array(input_label)})

            for g_epoch in tqdm(range(config.max_epochs_gen)):
                cnt = 0
                root_nodes = []  # just for record how many trees that have been traversed
                rel_nodes = []  # the sample nodes of the root node
                root_nodes_gen = []  # root node feeds into the network, same length as rel_node
                trace = []  # the trace when sampling the nodes, from the root to leaf  bach to leaf's father. e.g.: 0 - 1 - 2 -1
                all_score = self.sess.run(self.gen.all_score)  # compute the score for computing the probability when sampling nodes
                for root_node in tqdm(self.root_nodes, mininterval=3):  # random update trees
                    if np.random.rand() < config.update_ratio:
                        # sample the nodes according to our method.
                        # feed the reward from the discriminator and the sampled nodes to the gen.
                        if cnt % config.gen_update_iter == 0 and cnt > 0:
                            # generate update pairs along the path, [q_node, rel_node]
                            pairs = list(map(self.generate_window_pairs, trace))  # [[], []] each list contains the pairs along the same path
                            q_node_gen = []
                            rel_node_gen = []
                            for ii in range(len(pairs)):
                                path_pairs = pairs[ii]
                                for pair in path_pairs:
                                    q_node_gen.append(pair[0])
                                    rel_node_gen.append(pair[1])
                            reward_gen, node_embed = self.sess.run([self.dis.score, self.dis.node_embed],
                                                       {self.dis.q_node: np.array(q_node_gen),
                                                        self.dis.rel_node: np.array(rel_node_gen)})

                            feed_dict = {self.gen.q_node: np.array(q_node_gen), self.gen.rel_node: np.array(rel_node_gen),
                                         self.gen.reward: reward_gen}
                            _, loss, prob = self.sess.run([self.gen.gan_updates, self.gen.gan_loss, self.gen.i_prob],
                                                          feed_dict=feed_dict)
                            all_score = self.sess.run(self.gen.all_score)
                            root_nodes = []
                            rel_nodes = []
                            root_nodes_gen = []
                            trace = []
                            cnt = 0
                        sample = self.sample_for_gan(root_node, self.trees[root_node], config.n_sample_gen, all_score, sample_for_dis=False)
                        if len(sample) < config.n_sample_gen:
                            cnt = len(root_nodes)
                            continue
                        root_nodes.append(root_node)
                        root_nodes_gen.extend(len(sample)*[root_node])
                        rel_nodes.extend(sample)
                        trace.extend(self.trace)
                        cnt = cnt + 1

            print('Evaluation')
            self.write_emb_to_txt()
            self.eval_test()  # evaluation

    def generate_window_pairs(self, sample_path):
        sample_path = sample_path[:-1]
        pairs = []

        for i in range(len(sample_path)):
            center_node = sample_path[i]
            for j in range(max(i-config.window_size, 0), min(i+config.window_size+1, len(sample_path))):
                if i == j:
                    continue
                node = sample_path[j]
                pairs.append([center_node, node])

        return pairs

    def save_emb(self, node_embed, filename):
        np.savetxt(filename, node_embed, fmt='%10.5f', delimiter='\t')

    def write_emb_to_txt(self):
        modes = [self.gen, self.dis]
        for i in range(2):
            node_embed = self.sess.run(modes[i].node_embed)
            a = self.root_nodes.reshape(-1, 1)
            node_embed = np.hstack([a, node_embed])
            node_embed_list = node_embed.tolist()
            node_embed_str = ['\t'.join([str(x) for x in line]) + '\n' for line in node_embed_list]
            with open(config.emb_filenames[i], 'w+') as f:
                lines = [str(config.node_num) + '\t' + str(config.embed_dim) + '\n'] + node_embed_str
                f.writelines(lines)

    def eval_test(self):
        results = []
        for i in range(2):
            LPE = elp.LinkPredictEval(config.emb_filenames[i], config.test_filename, config.test_neg_filename, config.node_num, config.n_embed)
            result = LPE.eval_link_prediction()
            results.append(config.modes[i] + ':' + str(result) + '\n')
        with open(config.result_filename, mode='a+') as f:
            f.writelines(results)


if __name__ == '__main__':
    graphgan = GraphGan()
    graphgan.train()
