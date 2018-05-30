import tensorflow as tf


class Dis(object):
    def __init__(self, n_node, embed_dim, lamb, learning_rate):
        self.n_node = n_node
        with tf.variable_scope('discriminator'):
            self.node_embed = tf.get_variable(
                name='node_embed', shape=embed_dim,
                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                trainable=True)
            self.node_b = tf.Variable(tf.zeros([self.n_node]))

        self.q_node = tf.placeholder(tf.int32)
        self.rel_node = tf.placeholder(tf.int32)
        self.label = tf.placeholder(tf.float32)
        self.q_embedding = tf.nn.embedding_lookup(self.node_embed, self.q_node)
        self.rel_embedding = tf.nn.embedding_lookup(self.node_embed, self.rel_node)
        self.i_bias = tf.gather(self.node_b, self.rel_node)
        self.score = tf.reduce_sum(tf.multiply(self.q_embedding, self.rel_embedding), 1) + self.i_bias
        # prediction loss
        self.pre_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.score)) \
                        + lamb * (tf.nn.l2_loss(self.rel_embedding) + tf.nn.l2_loss(self.q_embedding) + tf.nn.l2_loss(self.i_bias))

        d_opt = tf.train.AdamOptimizer(learning_rate)
        self.d_updates = d_opt.minimize(self.pre_loss)
        # self.reward = config.reward_factor * (tf.sigmoid(self.score) - 0.5)
        self.score = tf.clip_by_value(self.score, clip_value_min=-10, clip_value_max=10)
        self.reward = tf.log(1 + tf.exp(self.score))
