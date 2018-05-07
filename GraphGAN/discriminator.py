# the discriminator class
import tensorflow as tf
import config


class Discriminator():
    def __init__(self, user_num, item_num, embed_init):
        self.user_num = user_num
        self.item_num = item_num
        self.embed_dim = config.emed_dim
        self.embed_init = embed_init
        self.init_delta = config.init_delta

        with tf.variable_scope('discriminator'):
            if self.embed_init == None:
                self.user_embeddings = tf.Variable(
                    tf.random_uniform([self.user_num, self.embed_dim], minval=-self.init_delta, maxval=self.init_delta,
                                      dtype=tf.float32))
                self.item_embeddings = tf.Variable(
                    tf.random_uniform([self.item_num, self.embed_dim], minval=-self.init_delta, maxval=self.init_delta,
                                      dtype=tf.float32))
                self.item_bias = tf.Variable(tf.zeros([self.item_num]))
            else:
                self.user_embeddings = tf.Variable(self.embed_init[0])
                self.item_embeddings = tf.Variable(self.embed_init[1])
                self.item_bias = tf.Variable(self.embed_init[2])

        self.u_nodes = tf.placeholder(tf.int32)
        self.i_nodes = tf.placeholder(tf.int32)
        self.label = tf.placeholder(tf.float32)
        self.u_embeddings = tf.nn.embedding_lookup(self.user_embeddings, self.u_nodes)
        self.i_embeddings = tf.nn.embedding_lookup(self.item_embeddings, self.i_nodes)
        self.i_bias = tf.gather(self.item_bias, self.i_nodes)
        self.score = tf.reduce_sum(tf.multiply(self.q_embedding, self.rel_embedding), 1) + self.i_bias
        # prediction loss
        self.pre_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.score)) \
                        + config.lambda_dis * (tf.nn.l2_loss(self.rel_embedding) + tf.nn.l2_loss(self.q_embedding) + tf.nn.l2_loss(self.i_bias))

        d_opt = tf.train.AdamOptimizer(config.lr_dis)
        self.d_updates = d_opt.minimize(self.pre_loss)
        # self.reward = config.reward_factor * (tf.sigmoid(self.score) - 0.5)
        self.score = tf.clip_by_value(self.score, clip_value_min=-10, clip_value_max=10)
        self.reward = tf.log(1 + tf.exp(self.score))
