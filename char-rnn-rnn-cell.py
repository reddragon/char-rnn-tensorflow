import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os.path
from tensorflow.models.rnn.ptb import reader

raw_data = open('input.txt', 'r').read()
# TODO Remove this later.
# raw_data = raw_data[0:50000]
chars = list(set(raw_data))
data_size, vocab_size = len(raw_data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# Convert an array of chars to array of vocab indices
def c2i(inp):
    return map(lambda c:char_to_ix[c], inp)

def i2c(inp):
    return map(lambda c:ix_to_char[c], inp)

data = c2i(raw_data)
'''
How PTB Iterator works

Seq Length: Number of inputs in each example.
Batch Size: Number of example rows in the batch.
Batch Len: Number of batches.
'''

def gen_epoch_data(num_epochs, batch_size, seq_length):
    for i in range(num_epochs):
        yield reader.ptb_iterator(data, batch_size, seq_length)


sample_data = range(1000)
num_epochs = 100
batch_size = 50
seq_length = 200
hidden_size = 100
learning_rate = 0.005
checkpoint_file = "rnn-cell-model.ckpt"

'''
for idx, epoch in enumerate(gen_epoch_data(num_epochs, batch_size, seq_length)):
    for X, Y in epoch:
        print X, Y
'''

def train_network(graph, num_epochs, batch_size, seq_length, checkpoint):
    tf.set_random_seed(2345)
    prev_epoch_loss = 1e50
    losses = []
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        if os.path.isfile(checkpoint):
            g['saver'].restore(sess, checkpoint)

        for idx, epoch in enumerate(gen_epoch_data(num_epochs, batch_size, seq_length)):
            # training_state = np.zeros([batch_size, hidden_size], dtype=float)
            training_state = None
            epoch_loss = 0.0
            batches = 0
            for batchIdx, (x, y) in enumerate(epoch):
                batches = batches + 1
                feed_dict = {
                    graph['x'] : x,
                    graph['y'] : y,
                }
                if training_state is not None:
                    feed_dict[graph['init_state']] = training_state

                _, _, rnn_inputs, rnn_outputs, init_state, training_state, \
                total_loss, train_step = sess.run(
                    [
                        graph['x_oh'],
                        graph['y_oh'],
                        graph['rnn_inputs'],
                        graph['rnn_outputs'],
                        graph['init_state'],
                        graph['final_state'],
                        graph['total_loss'],
                        graph['train_step'],
                    ],
                    feed_dict
                )
                epoch_loss += total_loss
                '''
                if batchIdx % 5 == 0:
                    print 'Epoch:', idx, 'Batch:', batchIdx, 'Loss:', total_loss
                    print init_state
                    print '---'
                    print training_state
                '''
            epoch_loss /= batches
            print 'Epoch:', idx, 'Average epoch loss:', epoch_loss
            if epoch_loss < prev_epoch_loss:
                g['saver'].save(sess, checkpoint_file)
            prev_epoch_loss = epoch_loss
            losses.append(epoch_loss)
    return losses

def build_graph(batch_size, seq_length, vocab_size, state_size, learning_rate):
    # Get the inputs.
    x = tf.placeholder(tf.int32, shape=([batch_size, seq_length]), name="x")
    y = tf.placeholder(tf.int32, shape=([batch_size, seq_length]), name="y")
    # init_state = tf.placeholder(tf.float32, shape=([batch_size, state_size]), name="init_state")

    # Converting x & y into one-hot representations.
    # x_oh & y_oh are of shape [batch_size, seq_length, vocab_size] now.
    x_oh = tf.one_hot(indices=x, depth=vocab_size)
    y_oh = tf.one_hot(indices=y, depth=vocab_size)

    # Basically converts the input into [seq_length, batch_size, vocab_size].
    rnn_inputs = tf.unpack(x_oh, axis=1)
    # y_oh is also of the same shape as x_oh
    # i.e., [seq_length, batch_size, vocab_size].
    y_as_list = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(1, seq_length, y)]

    # print tf.shape(x_oh), tf.shape(x)

    # Creates a hidden state of size state_size per batch.
    cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
    # init_state would be a vector of shape [batch_size, state_size].
    init_state = cell.zero_state(batch_size, tf.float32)

    # The RNN Cell abstracts away this calculation:
    # Hi = tanh(X Wxh + Hi-1 Whh + bh)
    # Wxh is of shape [vocab_size, hidden_size].
    # Whh is of shape [hidden_size, hidden_size].
    # The shape of the rnn_output would be [seq_length, batch_size, hidden_size].
    # The rnn() method will basically iterate over all the batches.
    # [[batch_size, vocab_size], [batch_size, vocab_size], ...].
    rnn_outputs, final_state = tf.nn.rnn(cell, rnn_inputs, initial_state=init_state)

    with tf.variable_scope('softmax'):
        Why = tf.get_variable('Why', [state_size, vocab_size])
        by = tf.get_variable('by', [vocab_size], initializer=tf.constant_initializer(0.0))

    logits = [tf.matmul(rnn_output, Why) + by for rnn_output in rnn_outputs]
    loss_weights = [tf.ones([batch_size]) for i in range(seq_length)]
    losses = tf.nn.seq2seq.sequence_loss_by_example(logits, y_as_list, loss_weights)
    total_loss = tf.reduce_mean(losses)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(
        x = x,
        y = y,
        x_oh = x_oh,
        y_oh = y_oh,
        rnn_inputs = rnn_inputs,
        rnn_outputs = rnn_outputs,
        init_state = init_state,
        final_state = final_state,
        total_loss = total_loss,
        train_step = train_step,
        saver = tf.train.Saver()
    )

g = build_graph(batch_size, seq_length, vocab_size, hidden_size, learning_rate)
losses = train_network(g, num_epochs, batch_size, seq_length, checkpoint_file)
f = open('rnn-cell-losses.txt', 'w')
for loss in losses:
    f.write(str(loss) + '\n')
f.close()
plt.plot(losses)
plt.show()
