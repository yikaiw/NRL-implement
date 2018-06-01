import tensorflow as tf


class Gen(object):
    def __init__(self, n_node, embed_dim, lamb, learning_rate):
        self.n_node = n_node
        with tf.variable_scope('Generator'):
            self.node_embed = tf.get_variable(
                name='node_embed', shape=(1, embed_dim),
                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                trainable=True)
            self.node_b = tf.Variable(tf.zeros([self.n_node]))

        self.all_score = tf.matmul(self.node_embed, self.node_embed, transpose_b=True) + self.node_b
        self.q_node = tf.placeholder(tf.int32, shape=[None])
        self.rel_node = tf.placeholder(tf.int32, shape=[None])
        self.reward = tf.placeholder(tf.float32, shape=[None])

        self.q_embedding= tf.nn.embedding_lookup(self.node_embed, self.q_node)  # batch_size*n_embed
        self.rel_embedding = tf.nn.embedding_lookup(self.node_embed, self.rel_node)  # batch_size*n_embed
        self.i_bias = tf.gather(self.node_b, self.rel_node)
        score = tf.reduce_sum(self.q_embedding*self.rel_embedding, axis=1) + self.i_bias
        i_prob = tf.nn.sigmoid(score)
        self.i_prob = tf.clip_by_value(i_prob, 1e-5, 1)
        self.gan_loss = -tf.reduce_mean(tf.log(self.i_prob) * self.reward) \
                        + lamb * (tf.nn.l2_loss(self.rel_embedding) + tf.nn.l2_loss(self.q_embedding))
        g_opt = tf.train.AdamOptimizer(learning_rate)
        self.gan_updates = g_opt.minimize(self.gan_loss)

