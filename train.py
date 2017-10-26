import time

#import tensorflow.python.platform
import collections
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf

from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn.rnn_cell import linear
from tensorflow.models.rnn.rnn_cell import RNNCell
from tensorflow.models.rnn import seq2seq
import sys
import logging
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
SEED=1234

class DomainDropoutWrapper(RNNCell):
  """Operator adding dropout to inputs and outputs of the given cell."""

  def __init__(self, cell, input_keep_prob=1.0, output_keep_prob=1.0,
               seed=None):
    """Create a cell with added input and/or output dropout.

    Dropout is never used on the state.

    Args:
      cell: an RNNCell, a projection to output_size is added to it.
      input_keep_prob: unit Tensor or float between 0 and 1, input keep
        probability; if it is float and 1, no input dropout will be added.
      output_keep_prob: unit Tensor or float between 0 and 1, output keep
        probability; if it is float and 1, no output dropout will be added.
      seed: (optional) integer, the randomness seed.

    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if keep_prob is not between 0 and 1.
    """
    if not isinstance(cell, RNNCell):
      raise TypeError("The parameter cell is not a RNNCell.")
    if (isinstance(input_keep_prob, float) and
        not (input_keep_prob >= 0.0 and input_keep_prob <= 1.0)):
      raise ValueError("Parameter input_keep_prob must be between 0 and 1: %d"
                       % input_keep_prob)
    if (isinstance(output_keep_prob, float) and
        not (output_keep_prob >= 0.0 and output_keep_prob <= 1.0)):
      raise ValueError("Parameter input_keep_prob must be between 0 and 1: %d"
                       % output_keep_prob)
    self._cell = cell
    self._input_keep_prob = input_keep_prob
    self._output_keep_prob = output_keep_prob
    self._seed = SEED

  @property
  def input_size(self):
    return self._cell.input_size

  @property
  def output_size(self):
    return self._cell.output_size

  @property
  def state_size(self):
    return self._cell.state_size

  def __call__(self, in_domain_inputs, out_domain_inputs, state, out_state, scope=None):
    """Run the cell with the declared dropouts."""
    if (not isinstance(self._input_keep_prob, float) or
        self._input_keep_prob < 1):
        in_domain_inputs = nn_ops.dropout(in_domain_inputs, self._input_keep_prob, seed=self._seed)
        out_domain_inputs = nn_ops.dropout(out_domain_inputs, self._input_keep_prob, seed=self._seed)
    output, new_state, new_out_state = self._cell(in_domain_inputs, out_domain_inputs, state, out_state)
    if (not isinstance(self._output_keep_prob, float) or
        self._output_keep_prob < 1):
      output = nn_ops.dropout(output, self._output_keep_prob, seed=self._seed)
    return output, new_state, new_out_state


class DomainGRUCell(RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self, num_units, input_size=None):
    self._num_units = num_units
    self._input_size = num_units if input_size is None else input_size

  @property
  def input_size(self):
    return self._input_size

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, in_domain_inputs, out_domain_inputs, indomain_state, outdomain_state, scope=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    #print "Gated recurrent unit (GRU) with nunits cells."
    with vs.variable_scope(scope or type(self).__name__):  # "GRUCell"


    with vs.variable_scope("domain-gate-input"):
        input_gate = linear([indomain_state, outdomain_state, in_domain_inputs, out_domain_inputs], 300, True, 1.0)
        

    with vs.variable_scope("domain-gate-state"):
        state_gate = linear([indomain_state, outdomain_state, in_domain_inputs, out_domain_inputs], 600, True, 1.0)
    
    with vs.variable_scope("domain-gate-apply"):
        input_gate, state_gate = sigmoid(input_gate), sigmoid(state_gate)

        in_domain_inputs = input_gate * in_domain_inputs + (1-input_gate) * out_domain_inputs
        indomain_state = state_gate * indomain_state + (1-state_gate) * outdomain_state



        with vs.variable_scope("in-domain"):  # Reset gate and update gate.
        
            in_r, in_u = array_ops.split(1, 2, linear([in_domain_inputs, indomain_state],
                                            2 * self._num_units, True, 1.0))
            in_r, in_u = sigmoid(in_r), sigmoid(in_u)


    with vs.variable_scope("out-domain"):  # Reset gate and update gate.
        
            out_r, out_u = array_ops.split(1, 2, linear([out_domain_inputs, outdomain_state],
                                            2 * self._num_units, True, 1.0))
            out_r, out_u = sigmoid(out_r), sigmoid(out_u)

    with vs.variable_scope("out-Candidate"):
        c_out = tanh(linear([out_domain_inputs, out_r * outdomain_state], self._num_units, True))

    with vs.variable_scope("in-Candidate"):
        c_in = tanh(linear([in_domain_inputs, in_r * indomain_state], self._num_units, True))

    
        new_h = in_u * indomain_state + (1 - in_u) * c_in
    new_h_d = out_u * outdomain_state + (1 - out_u) * c_out
    return new_h, new_h, new_h_d


class loggerinitializer(object):

    @staticmethod
    def initialize_logger(output_dir):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        # create console handler and set level to info
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s;%(levelname)s\t%(message)s","%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # create error file handler and set level to error
        handler = logging.FileHandler(output_dir,"w", encoding=None, delay="true")
        handler.setLevel(logging.ERROR)
        formatter = logging.Formatter("%(asctime)s;%(levelname)s\t%(message)s","%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # create debug file handler and set level to debug
        handler = logging.FileHandler(output_dir,"w")
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s;%(levelname)s\t%(message)s", "%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

class PTBModel(object):
    """The PTB model."""

    def __init__(self, is_training, vocab_size, hidden_size, batch_size=20, num_steps=40,
                 keep_prob=0.5, max_grad_norm = 5, num_layers=1):
        self.batch_size = batch_size
        self.num_steps = num_steps = num_steps
        size = hidden_size
        vocab_size = vocab_size

        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])
    with tf.device("/cpu:0"):        
            self._g_embedding_W = tf.Variable(tf.constant(0.0, shape=[vocab_size, 300]),
                trainable=False, name="g_embedding_W")

            self._g_embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, 300])
            self._g_embedding_init = self._g_embedding_W.assign(self._g_embedding_placeholder)
        
        
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, 300])
            inputs_i = tf.nn.embedding_lookup(embedding, self._input_data)
        if is_training and keep_prob < 1:
            inputs_i = tf.nn.dropout(inputs_i, keep_prob)
        
        
          
        lstm_cell = DomainGRUCell(num_units=size)
        if is_training and keep_prob < 1:
            lstm_cell = DomainDropoutWrapper(
                    lstm_cell, output_keep_prob=keep_prob, seed=SEED)        

        self._initial_state = lstm_cell.zero_state(batch_size, tf.float32)
    self._initial_state_domain = lstm_cell.zero_state(batch_size, tf.float32)

        with tf.device("/cpu:0"):
            inputs_g = tf.nn.embedding_lookup(self._g_embedding_W, self._input_data)
        #if is_training and keep_prob < 1:
        #    inputs_g = tf.nn.dropout(inputs_g, keep_prob)
        
        outputs = []
        state = self._initial_state
    state_domain = self._initial_state_domain
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state, state_domain) = lstm_cell(inputs_i[:, time_step, :], inputs_g[:, time_step, :], state, state_domain)
                outputs.append(cell_output)
        
        
     
        long_vect = tf.concat(1, outputs)
        output = tf.reshape(long_vect, [-1, size])
        
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])        
        logits = tf.matmul(output, softmax_w) + softmax_b
        
        
        loss = seq2seq.sequence_loss_by_example([logits],
            [tf.reshape(self._targets, [-1])],
            [tf.ones([batch_size * num_steps])],vocab_size)
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state
    self._final_state_domain = state_domain

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        for v in tvars:
            logging.info('Variable: ' + v.name)
                
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    @property
    def g_embedding_W(self):
        return self._g_embedding_W
    
    @property
    def g_embedding_placeholder(self):
        return self._g_embedding_placeholder
    
    
    @property
    def g_embedding_init(self):
        return self._g_embedding_init
    
    
    @property
    def input_data(self):
        return self._input_data

    
    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def initial_state_domain(self):
        return self._initial_state_domain

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def final_state_domain(self):
        return self._final_state_domain

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

def ptb_iterator(raw_data, batch_size, num_steps):

    raw_data = np.array(raw_data, dtype=np.int32)
    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)

    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x, y)



def run_epoch(session, m, data, eval_op, verbose=False):
    """Runs the model on the given data."""
    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    costs = 0.0
    iters = 0
    state = m.initial_state.eval()
    state_domain = m.initial_state_domain.eval()
    counter = 0
    for step, (x, y) in enumerate(ptb_iterator(data, m.batch_size, m.num_steps)):

        
        start = time.time()
        
        cost, state, state_domain, _ = session.run([m.cost, m.final_state, m.final_state_domain, eval_op],
                                                                 {m.input_data: x,
                                                                    m.targets: y,
                                                                   m.initial_state: state,
                                  m.initial_state_domain: state_domain})
        
        counter+=1

        #if counter == 1:
        #    logging.info("takes %f seconds, and will takes about %f seconds for each eopch" % (time.time() - start, epoch_size * (time.time() - start)))
        costs += cost
        iters += m.num_steps
        
    return np.exp(costs / iters)

def _read_words(filename, vocab_size, max_line=0):
    with gfile.GFile(filename, "r") as f:
        if max_line == 0:
            data = f.read().split()
        else:
            data = f.read().split()[0:max_line]
    
    unk = [word if word < vocab_size else 0 for word in map(int, data)]
    return unk

    
def ptb_raw_data(train_file,\
                valid_file, \
                test_file,vocab_size):

    train_data = _read_words(train_file, vocab_size)
    logging.info( "read : train_data")
    valid_data = _read_words(valid_file, vocab_size)
    logging.info( "read : valid_data")
    test_data = _read_words(test_file, vocab_size)
    logging.info( "read : test_data")

    return train_data, valid_data, test_data
    


def get_pretrain_embedding(file, vocab_size_g):
    emb = []
    with  open(file, 'r') as f:
        for line in f:
            num = (line.strip().split('\t')[2])
            emb.append(np.array(num.split(','), dtype='float32'))
    emb = np.array(emb)
    return emb[0:vocab_size_g]

    

def main(train_file, valid_file, test_file,
         saveto, vocab_size, w2v,
         init_scale, max_max_epoch, lr_decay, max_epoch, hidden_size,
         learning_rate=1.0):

    #train_data, valid_data, test_data  = ptb_raw_data(train_file, \
    #                                                        valid_file, \
    #                                                        test_file, vocab_size)
    tf.set_random_seed(SEED)
    with tf.Graph().as_default(), tf.Session() as session:
    
        initializer = tf.random_uniform_initializer(-init_scale,
                        init_scale, seed = SEED)
        
        pretrain_embedding = get_pretrain_embedding(w2v, vocab_size)
        g_vocab_size = pretrain_embedding.shape[0]
        g_dim_size = pretrain_embedding.shape[1]
        
        logging.info( "g_vocab_size %d " % g_vocab_size)
        logging.info( "g_dim_size %d " % g_dim_size)
        '''
        for step, (x, y) in enumerate(ptb_iterator(train_data, 1, 1)):
        for t in x:
        if (np.sum(pretrain_embedding[t]) == 0):
            print pretrain_embedding[t]
            print t
            print "error"
                sys.exit()         
        '''

        logging.info( "initializer ok")
       
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = PTBModel(is_training=True, vocab_size=vocab_size, hidden_size=hidden_size)

        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mvalid = PTBModel(is_training=False, vocab_size=vocab_size, hidden_size=hidden_size)
            mtest = PTBModel(is_training=False,  batch_size = 1, num_steps = 1, vocab_size=vocab_size, hidden_size=hidden_size)

    
        #tf.initialize_all_variables().run()
        session.run(m.g_embedding_init, feed_dict={m.g_embedding_placeholder: pretrain_embedding})
    session.run(mvalid.g_embedding_init, feed_dict={mvalid.g_embedding_placeholder: pretrain_embedding})
    session.run(mtest.g_embedding_init, feed_dict={mtest.g_embedding_placeholder: pretrain_embedding})
        tf.initialize_all_variables().run()
        
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            print(shape)
            print(len(shape))
            variable_parametes = 1
            for dim in shape:
                print(dim)
                variable_parametes *= dim.value
            print(variable_parametes)
            total_parameters += variable_parametes
        print(total_parameters)      
        sys.exit()  
        for i in range(max_max_epoch):
            new_lr_decay = lr_decay ** max(i - max_epoch, 0.0)
            m.assign_lr(session, learning_rate * new_lr_decay)

            logging.info("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            train_perplexity = run_epoch(session, m, train_data, m.train_op,verbose=True)
            logging.info("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

            valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
            logging.info("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
            logging.info("Epoch: %d Test Perplexity: %.3f" % (i + 1, test_perplexity))

            saver = tf.train.Saver()
            save_path = saver.save(session, saveto + '/rnn_iter' + str(i) + ".model")
            logging.info("Model saved in file: %s" % save_path)


if __name__ == "__main__":
    np.set_printoptions(threshold=np.nan)
    
    train_file = sys.argv[1]
    valid_file = sys.argv[2]
    test_file = sys.argv[3]
    w2v= sys.argv[4]
    saveto = sys.argv[5]
    vocab_size = int(sys.argv[6])
    log = sys.argv[7]
    init_scale = float(sys.argv[8])
    max_max_epoch = int(sys.argv[9])
    lr_decay = float(sys.argv[10])
    max_epoch = int(sys.argv[11])
    hidden_size = int(sys.argv[12])

    loggerinitializer.initialize_logger(log)
    for a in sys.argv[1:]:
    logging.info(str(a))

    main(train_file, valid_file, test_file,saveto, vocab_size, w2v, init_scale, max_max_epoch, lr_decay, max_epoch, hidden_size)
