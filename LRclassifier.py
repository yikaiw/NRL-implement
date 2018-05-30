import numpy as np
import tensorflow as tf
import util_cora

training_epochs = 200
batch_size = 16
embed_dim = 128
class_dim = 7
learning_rate = tf.train.exponential_decay(
    learning_rate=0.1, global_step=10000, decay_steps=100, decay_rate=0.98, staircase=True)

method = 'node2vec'  # DeepWalk, LINE, node2vec
data = util_cora.Dataset(method)
if method == 'LINE':
    embed_dim *= 2
X = tf.placeholder(tf.float32, [None, embed_dim])
y = tf.placeholder(tf.int32, [None, ])
labels = tf.one_hot(y, class_dim, axis=1)
W = tf.Variable(tf.zeros([embed_dim, class_dim]))
b = tf.Variable(tf.zeros([class_dim]))

logits = tf.nn.softmax(tf.matmul(X, W) + b)
loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    for epoch in range(training_epochs):
        avg_cost = 0
        train_len = len(data.train_y)
        permutation = np.random.permutation(train_len)
        train_X = data.train_X[permutation]
        train_y = data.train_y[permutation]
        data_idx = 0
        while data_idx < train_len - 1:
            X_batch = train_X[data_idx: np.clip(data_idx + batch_size, 0, train_len - 1)]
            y_batch = train_y[data_idx: np.clip(data_idx + batch_size, 0, train_len - 1)]
            data_idx += batch_size
            batch_loss, _ = sess.run([loss, optimizer], feed_dict={X: X_batch, y: y_batch})
            if step == 0:
                print('Step %5d: initial loss = %.5f' % (step, batch_loss))
            step += 1
            if step % 10 == 0:
                print('\rStep %5d: loss = %.5f' % (step, batch_loss), end='')

    print("\nOptimization Finished.")

    # Test model
    correct_pre = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))
    print("Accuracy:", accuracy.eval({X: data.test_X, y: data.test_y}))
