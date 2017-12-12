#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os

import tensorflow as tf
import scipy.io.wavfile as wav
import numpy as np
from scipy import signal
from sklearn import preprocessing
from scipy.signal import resample_poly

from six.moves import xrange as range

from python_speech_features import mfcc, sigproc, delta
from utils import variable_on_cpu
from utils import sparse_tuple_from as sparse_tuple_from
from utils import pad_sequences as pad_sequences
from glob import glob
from DataGenerator import DataGenerator, _input_data
from text import ndarray_to_text, sparse_tuple_to_texts





num_classes = 29
num_features = 513



# Hyper-parameters
num_epochs = 200
num_hidden = 150
num_layers = 1
batch_size = 2
initial_learning_rate = 1e-2
momentum = 0.9





def _placeholder(num_features):
        # e.g: log filter bank or MFCC features
        # Has size [batch_size, max_stepsize, num_features], but the
        # batch_size and max_stepsize can vary along each step
        inputs = tf.placeholder(tf.float32, [None, None, num_features])


        # Here we use sparse_placeholder that will generate a
        # SparseTensor required by ctc_loss op.
        targets = tf.sparse_placeholder(tf.int32)

        # 1d array of size [batch_size]
        seq_len = tf.placeholder(tf.int32, [None])


        return inputs, targets, seq_len


def _network(inputs, seq_len):

    dropout = [0.05, 0.05, 0.05, 0.05, 0.01,0.05]
    relu_clip = 20
    ## RRN layers
    shape = tf.shape(inputs)
    batch_s, max_timesteps = shape[0], shape[1]
    batch_x = tf.transpose(inputs,[1,0,2])
    cell_fw = tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden, state_is_tuple=True)
    cell_bw = tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden, state_is_tuple=True)

    outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                      cell_bw=cell_bw,
                                                      inputs = batch_x,
                                                      sequence_length=seq_len,
                                                      time_major=True,
                                                      dtype=tf.float32)



    # Reshaping to apply the same weights over the timesteps
    #outputs = tf.reshape(outputs, [-1, num_hidden])
    outputs = tf.concat(outputs,2)
    #outputs = simple_attention(outputs,2*num_hidden, time_major=True,return_alphas=False)
    outputs = tf.reshape(outputs, [-1, 2*num_hidden])

    #layer_3= tf.contrib.layers.fully_connected(inputs=outputs, num_outputs=4*num_hidden, activation_fn=tf.nn.relu)
    #logits= tf.contrib.layers.fully_connected(inputs=layer_3, num_outputs=num_classes, activation_fn=None)
    # Reshaping back to the original shape
    with tf.name_scope('fc5'):
        # Now we feed `outputs` to the fifth hidden layer with clipped RELU activation and dropout
        b5 = variable_on_cpu('b5', [4*num_hidden], tf.random_normal_initializer(stddev=0.046875))
        h5 = variable_on_cpu('h5', [(2*num_hidden), 4*num_hidden], tf.random_normal_initializer(stddev=0.046875))
        layer_5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs, h5), b5)), relu_clip)
        layer_5 = tf.nn.dropout(layer_5, (1.0 - dropout[5]))

    with tf.name_scope('fc6'):

        b6 = variable_on_cpu('b6', [num_classes], tf.random_normal_initializer(stddev=0.046875))
        h6 = variable_on_cpu('h6', [4*num_hidden, num_classes], tf.random_normal_initializer(stddev=0.046875))
        logits = tf.add(tf.matmul(layer_5, h6), b6)


    logits = tf.reshape(logits, [-1, batch_s, num_classes])

    return logits






# THE MAIN CODE!

graph = tf.Graph()
with graph.as_default():

    ## Placeholders
    inputs, targets, seq_len = _placeholder(num_features)
    ## Networks Gemotry
    logits = _network(inputs, seq_len)

    ## Loss
    loss = tf.nn.ctc_loss(targets, logits, seq_len)
    cost = tf.reduce_mean(loss)

    ## Optimizer
    optimizer = tf.train.MomentumOptimizer(learning_rate = initial_learning_rate,
                                           momentum = 0.9,
                                           use_nesterov=True).minimize(cost)

    # Option 2: tf.nn.ctc_beam_search_decoder
    # (it's slower but you'll get better results)
    decoded, log_prob =tf.nn.ctc_greedy_decoder(logits, seq_len)
    #decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, output_lengths)

    # Inaccuracy: label error rate
    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                          targets))






# Dataset directories
training_dir = os.path.join(os.getcwd(),'data','training')
validation_dir = os.path.join(os.getcwd(),'data','validation')
testing_dir = os.path.join(os.getcwd(),'data','testing')


# Batching the data
trainnig_data = DataGenerator(training_dir, batch_size=7)
validation_data = DataGenerator(validation_dir, batch_size=7)

num_examples = len(trainnig_data) # number of files in training dataset

### Training of the network
with tf.Session(graph=graph) as session:
    # Initializate the weights and biases
    tf.global_variables_initializer().run()


    for curr_epoch in range(num_epochs):
        train_cost = train_ler = 0
        start = time.time()
        ## Training of the network
        for (train_inputs, train_targets, train_seq_len) in trainnig_data.next_batch():

            feed = {inputs: train_inputs,
                    targets: train_targets,
                    seq_len: train_seq_len}

            batch_cost, _ = session.run([cost, optimizer], feed)
            train_cost += batch_cost*train_inputs.shape[0]
            train_ler += session.run(ler, feed_dict=feed)*batch_size

        train_cost /= num_examples
        train_ler /= num_examples

        # Validation of the network
        for (val_inputs, val_targets, val_seq_len) in validation_data.next_batch():
            val_feed = {inputs: val_inputs,
                        targets: val_targets,
                        seq_len: val_seq_len}

            val_cost, val_ler = session.run([cost, ler], feed_dict=val_feed)

            log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"
            print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler,
                            val_cost, val_ler, time.time() - start))



    # Testing the system
    testing_data = DataGenerator(testing_dir, batch_size=1)
    for (test_inputs, test_targets, test_seq_len) in testing_data.next_batch():
        feed = {inputs: test_inputs,
                targets: test_targets,
                seq_len: test_seq_len}

        # Decoding
        d = session.run(decoded[0], feed_dict=feed)
        dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=session)
        dense_labels = sparse_tuple_to_texts(test_targets)

        for orig, decoded_arr in zip(dense_labels, dense_decoded):
            str_decoded = ndarray_to_text(decoded_arr)
            print('Original: {}'.format(orig))
            print('Decoded: {}'.format(str_decoded))
