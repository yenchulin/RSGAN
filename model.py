# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to build and run the tensorflow graph for the sequence-to-sequence model"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from attention_decoder import my_attention_decoder

FLAGS = tf.app.flags.FLAGS


def sample_output(embedding, embedding_dec, output_projection=None,
                               given_number=None):
  """Get a loop_function that extracts the previous symbol and embeds it.

  Args:
    embedding: embedding tensor for symbols.
    output_projection: None or a pair (W, B). If provided, each fed previous
      output will first be multiplied by W and added B.
    update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.

  Returns:
    A loop function.
  """
  def loop_function(prev,_):

    prev = tf.nn.xw_plus_b(
          prev, output_projection[0], output_projection[1])
    prev_symbol = tf.cast(tf.reshape(tf.multinomial(prev, 1), [FLAGS.batch_size*FLAGS.max_dec_sen_num]), tf.int32)
    # TODO: ensure tf.cast is doing condional sampling or sth else.
    emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
    return emb_prev # word embedding

  def loop_function_max(prev,_):
      """function that feed previous model output rather than ground truth."""
      if output_projection is not None:
          prev = tf.nn.xw_plus_b(
              prev, output_projection[0], output_projection[1])
      prev_symbol = tf.argmax(prev, 1)
      emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
      #emb_prev = tf.stop_gradient(emb_prev)
      return emb_prev

  '''def f1(prev,i):
      prev = tf.nn.xw_plus_b(
          prev, output_projection[0], output_projection[1])
      prev_symbol = tf.cast(tf.reshape(tf.multinomial(prev, 1), [FLAGS.batch_size]), tf.int32)
      emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
      return  emb_prev
  def f2(prev,i):
      emb_prev = embedding_dec[i]
      return emb_prev'''

  def loop_given_function(prev, i):

      return tf.cond(tf.less(i,2), lambda :loop_function(prev,i), lambda:loop_function_max(prev,i))


  return loop_function,loop_function_max,loop_given_function

class Generator(object):


  def __init__(self, hps, vocab):
    self._hps = hps
    self._vocab = vocab

  def _add_placeholders(self):
    """Add placeholders to the graph. These are entry points for any input data."""
    hps = self._hps

    if FLAGS.run_method == 'auto-encoder':
        self._enc_aspect_batch = tf.placeholder(tf.float32, [hps.batch_size.value, hps.max_dec_sen_num.value, None, hps.aspect_dim.value], name='enc_aspect_batch')
        self._enc_sentiment_batch = tf.placeholder(tf.float32, [hps.batch_size.value, None, hps.sentiment_dim.value], name='enc_sentiment_batch')
        self._enc_batch = tf.placeholder(tf.int32, [hps.batch_size.value, None], name='enc_batch')
        self._enc_lens = tf.placeholder(tf.int32, [hps.batch_size.value], name='enc_lens')
        #self._enc_padding_mask = tf.placeholder(tf.float32, [hps.batch_size.value, None], name='enc_padding_mask')

    self._AEdec_batch = tf.placeholder(tf.int32, [hps.batch_size.value, hps.max_enc_steps.value], name='AEdec_batch')
    self._AEdec_padding_mask = tf.placeholder(tf.float32, [hps.batch_size.value, hps.max_enc_steps.value], name='AEdec_padding_mask')

    self._dec_batch = tf.placeholder(tf.int32, [hps.batch_size.value, hps.max_dec_sen_num.value, hps.max_dec_steps.value], name='dec_batch')
    self._target_batch = tf.placeholder(tf.int32, [hps.batch_size.value*hps.max_dec_sen_num.value, hps.max_dec_steps.value], name='target_batch')
    self._dec_padding_mask = tf.placeholder(tf.float32, [hps.batch_size.value*hps.max_dec_sen_num.value, hps.max_dec_steps.value], name='dec_padding_mask')
    self.reward = tf.placeholder(tf.float32, [hps.batch_size.value*hps.max_dec_sen_num.value, hps.max_dec_steps.value], name='reward')
    self.dec_lens = tf.placeholder(tf.int32, [hps.batch_size.value], name='dec_lens')


  def _make_feed_dict(self, batch, just_enc=False):

    feed_dict = {}

    if FLAGS.run_method == 'auto-encoder':
        feed_dict[self._enc_aspect_batch] = batch.enc_aspect_batch
        feed_dict[self._enc_sentiment_batch] = batch.enc_sentiment_batch
        feed_dict[self._enc_batch] = batch.enc_batch
        feed_dict[self._enc_lens] = batch.enc_lens
        #feed_dict[self._enc_padding_mask] = batch.enc_padding_mask

    feed_dict[self._AEdec_batch] = batch.AEdec_batch
    feed_dict[self._AEdec_padding_mask] = batch.AEdec_padding_mask

    feed_dict[self._dec_batch] = batch.dec_batch
    feed_dict[self._target_batch] = batch.target_batch
    feed_dict[self._dec_padding_mask] = batch.dec_padding_mask
    feed_dict[self.dec_lens] = batch.dec_lens
    return feed_dict



  def _add_encoder(self, encoder_inputs, seq_len):

    with tf.variable_scope('encoder'):
      cell_fw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim.value, initializer=self.rand_unif_init, state_is_tuple=True)
      cell_bw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim.value, initializer=self.rand_unif_init, state_is_tuple=True)
      ((encoder_outputs_forward, encoder_outputs_backward), (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs, dtype=tf.float32, sequence_length=seq_len, swap_memory=True)
    return fw_st, bw_st, tf.concat([encoder_outputs_forward, encoder_outputs_backward],axis=-1)

  def _reduce_states(self, fw_st, bw_st):
    """Add to the graph a linear layer to reduce the encoder's final FW and BW state into a single initial state for the decoder. This is needed because the encoder is bidirectional but the decoder is not.

    Args:
      fw_st: LSTMStateTuple with hidden_dim units.
      bw_st: LSTMStateTuple with hidden_dim units.

    Returns:
      state: LSTMStateTuple with hidden_dim units.
    """
    hidden_dim = self._hps.hidden_dim.value
    with tf.variable_scope('reduce_final_st'):

      # Define weights and biases to reduce the cell and reduce the state
      w_reduce_c = tf.get_variable('w_reduce_c', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      w_reduce_h = tf.get_variable('w_reduce_h', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)

      # Apply linear layer
      old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c]) # Concatenation of fw and bw cell
      old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h]) # Concatenation of fw and bw state
      new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c) # Get new cell from old cell
      new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h) # Get new state from old state
      return tf.contrib.rnn.LSTMStateTuple(new_c, new_h) # Return new cell and state


  def _add_LMdecoder(self, loop_function, loop_function_max, loop_given_function, input, attention_state, aspect_feature, sentiment_feature):  # input batch sequence dim

    hps = self._hps



    #input = tf.unstack(input, axis=1)

    input = tf.reshape(input, [hps.batch_size.value*hps.max_dec_sen_num.value, hps.max_dec_steps.value , hps.emb_dim.value]) # (B*S, T, E)
    input = tf.unstack(input, axis = 1) # (B*S, E) * T, list of tensor

    cell = tf.contrib.rnn.LSTMCell(
      hps.hidden_dim.value,
      initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113),
      state_is_tuple=True)



    decoder_outputs_pretrain,_ = my_attention_decoder(
      input, self._dec_in_state,attention_state,
      cell, loop_function=None, aspect_feature=aspect_feature, sentiment_feature=sentiment_feature
    )


    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        decoder_outputs_sample_generator,_ = my_attention_decoder(
            input, self._dec_in_state,attention_state,
            cell, loop_function=loop_function, aspect_feature=aspect_feature, sentiment_feature=sentiment_feature
        )

        decoder_outputs_max_generator, _ = my_attention_decoder(
            input, self._dec_in_state,attention_state,
            cell, loop_function=loop_function_max, aspect_feature=aspect_feature, sentiment_feature=sentiment_feature
        )

        decoder_outputs_given_sample_generator, _ = my_attention_decoder(
            input, self._dec_in_state,attention_state,
            cell, loop_function=loop_given_function, aspect_feature=aspect_feature, sentiment_feature=sentiment_feature
        )

        '''decoder_outputs_generator_rollout = tf.contrib.legacy_seq2seq.rnn_decoder(
            input, self._dec_in_state,
            cell, loop_function=loop_given_function
        )'''




        decoder_outputs_pretrain = tf.stack(decoder_outputs_pretrain, axis=1)
        decoder_outputs_sample_generator = tf.stack(decoder_outputs_sample_generator, axis=1)
        decoder_outputs_max_generator = tf.stack(decoder_outputs_max_generator, axis=1)
        decoder_outputs_given_sample_generator = tf.stack(decoder_outputs_given_sample_generator, axis=1)
        #decoder_outputs_generator_rollout = tf.transpose(decoder_outputs_generator_rollout, [1, 0, 2])

    return decoder_outputs_pretrain, decoder_outputs_sample_generator, decoder_outputs_max_generator,decoder_outputs_given_sample_generator

  def _add_AEdecoder(self, input, attention_state, aspect_feature, sentiment_feature):  # input batch sequence dim
    hps = self._hps
    
    input = tf.unstack(input, axis = 1) # (B, E) * T, list of tensor

    cell = tf.contrib.rnn.LSTMCell(
      hps.hidden_dim.value,
      initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113),
      state_is_tuple=True)

    decoder_outputs_pretrain, _ = my_attention_decoder(
      input, self._dec_in_state, attention_state,
      cell, loop_function=None, aspect_feature=aspect_feature, sentiment_feature=sentiment_feature
    )
    return decoder_outputs_pretrain

  def _build_model(self):
    """Add the whole generator model to the graph."""
    hps = self._hps
    vsize = self._vocab.size() # size of the vocabulary

    with tf.variable_scope('seq2seq'):
      # Some initializers
      self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag.value, hps.rand_unif_init_mag.value, seed=123)
      self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std.value)

      # Add embedding matrix (shared by the encoder and decoder inputs)
      with tf.variable_scope('embedding'):
        embedding = tf.get_variable('embedding', [vsize, hps.emb_dim.value], dtype=tf.float32, initializer=self.trunc_norm_init)
        emb_dec_inputs = tf.nn.embedding_lookup(embedding, self._dec_batch) # tensor with shape (batch_size, max_dec_sen_num, max_dec_steps, emb_size)
        AEemb_dec_inputs = tf.nn.embedding_lookup(embedding, self._AEdec_batch) # tensor with shape (batch_size, max_enc_steps, emb_size)
        if FLAGS.run_method == 'auto-encoder':
            emb_enc_inputs = tf.nn.embedding_lookup(embedding, self._enc_batch)  # tensor with shape (batch_size, max_enc_steps, emb_size)
            fw_st, bw_st, encoder_outputs_word = self._add_encoder(emb_enc_inputs, self._enc_lens)
            self._dec_in_state = self._reduce_states(fw_st, bw_st)     

      if FLAGS.auto_encoder:
        with tf.variable_scope('AE_decoder'):
          with tf.variable_scope('output_projection'):
            w = tf.get_variable(
              'w', [hps.hidden_dim.value, vsize], dtype=tf.float32,
              initializer=tf.truncated_normal_initializer(stddev=1e-4))
            v = tf.get_variable(
              'v', [vsize], dtype=tf.float32,
              initializer=tf.truncated_normal_initializer(stddev=1e-4))
          
          AEdecoder_outputs_pretrain = self._add_AEdecoder(input=AEemb_dec_inputs, attention_state=encoder_outputs_word, aspect_feature=None, sentiment_feature=None)
          AEdecoder_outputs_pretrain = tf.reshape(AEdecoder_outputs_pretrain, [hps.batch_size.value * hps.max_enc_steps.value, hps.hidden_dim.value])
          AEdecoder_outputs_pretrain = tf.nn.xw_plus_b(AEdecoder_outputs_pretrain, w, v)
          AEdecoder_outputs_pretrain = tf.reshape(AEdecoder_outputs_pretrain, [hps.batch_size.value, hps.max_enc_steps.value, vsize])

          # Pad encoder input to max_enc_steps
          enc_batch = self._enc_batch
          diff = hps.max_enc_steps.value - self._enc_batch.shape[1]
          if diff > 0:
            paddings = tf.constant([[0, 0], [diff, 0]])
            enc_batch = tf.pad(enc_batch, paddings, 'CONSTANT')
            print(enc_batch.shape)

          AEloss = tf.contrib.seq2seq.sequence_loss(
              AEdecoder_outputs_pretrain,
              enc_batch,
              self._AEdec_padding_mask,
              average_across_timesteps=True,
              average_across_batch=False)

          AEreward_loss = tf.contrib.seq2seq.sequence_loss(
              AEdecoder_outputs_pretrain,
              enc_batch,
              self._AEdec_padding_mask,
              average_across_timesteps=True,
              average_across_batch=False)

          # Update the cost
          self._AEcost = tf.reduce_mean(AEloss)
          self._AEreward_cost = tf.reduce_mean(AEreward_loss)
          self.AEoptimizer = tf.train.AdagradOptimizer(self._hps.lr.value, initial_accumulator_value=self._hps.adagrad_init_acc.value)

      with tf.variable_scope('LM_decoder'):
        # Word-level encoder output
        encoder_outputs_word = tf.reshape(
            tf.tile(tf.expand_dims(encoder_outputs_word, axis=1), [1, hps.max_dec_sen_num.value,1, 1]),
            [hps.batch_size.value* hps.max_dec_sen_num.value, -1, hps.hidden_dim.value*2])
        enc_aspect_batch = tf.reshape(self._enc_aspect_batch,
            [hps.batch_size.value* hps.max_dec_sen_num.value, -1, hps.aspect_dim.value]
        ) if FLAGS.aspect_attn else None
        enc_sentiment_batch = tf.reshape(
            tf.tile(tf.expand_dims(self._enc_sentiment_batch, axis=1), [1, hps.max_dec_sen_num.value, 1, 1]),
            [hps.batch_size.value* hps.max_dec_sen_num.value, -1, hps.sentiment_dim.value]
        ) if FLAGS.senti_attn else None
        
        # Sentence-level encoder output
        sentence_level_input = tf.reshape(tf.tile(tf.expand_dims(self._dec_in_state.h, axis=1), [1, hps.max_dec_sen_num.value, 1]), [hps.batch_size.value, hps.max_dec_sen_num.value, hps.hidden_dim.value])
        sentence_level_cell = tf.contrib.rnn.LSTMCell(hps.hidden_dim.value, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113), state_is_tuple=True)
        (encoder_outputs, _) = tf.nn.dynamic_rnn(sentence_level_cell, sentence_level_input, dtype=tf.float32, sequence_length=self.dec_lens, swap_memory=True)
        encoder_outputs = tf.reshape(encoder_outputs, [hps.batch_size.value * hps.max_dec_sen_num.value, hps.hidden_dim.value])
        self._dec_in_state =  tf.contrib.rnn.LSTMStateTuple(encoder_outputs, encoder_outputs)

        with tf.variable_scope('output_projection'):
          w = tf.get_variable(
            'w', [hps.hidden_dim.value, vsize], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=1e-4))
          v = tf.get_variable(
            'v', [vsize], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=1e-4))

        # Add the decoder.
        loop_function, loop_function_max,loop_given_function = sample_output(embedding, emb_dec_inputs, (w, v))

        # 4 different decoders (loop function is different, loop function defines the relation between current decoder output and next decoder input)
        # the following code are doing reshape and add a linear layer + bias for the 4 decoder
        decoder_outputs_pretrain, decoder_outputs_sample_generator, decoder_outputs_max_generator, decoder_outputs_given_sample_generator = self._add_LMdecoder(loop_function=loop_function, loop_function_max=loop_function_max, loop_given_function=loop_given_function, input=emb_dec_inputs, attention_state=encoder_outputs_word, aspect_feature=enc_aspect_batch, sentiment_feature=enc_sentiment_batch)
        decoder_outputs_pretrain = tf.reshape(decoder_outputs_pretrain, [hps.batch_size.value * hps.max_dec_sen_num.value * hps.max_dec_steps.value, hps.hidden_dim.value])
        decoder_outputs_pretrain = tf.nn.xw_plus_b(decoder_outputs_pretrain, w, v)
        decoder_outputs_pretrain = tf.reshape(decoder_outputs_pretrain, [hps.batch_size.value * hps.max_dec_sen_num.value, hps.max_dec_steps.value, vsize])

        decoder_outputs_sample_generator = tf.reshape(decoder_outputs_sample_generator, [hps.batch_size.value * hps.max_dec_sen_num.value * hps.max_dec_steps.value, hps.hidden_dim.value])
        decoder_outputs_sample_generator = tf.nn.xw_plus_b(decoder_outputs_sample_generator, w, v)
        self._sample_best_output = tf.reshape(tf.argmax(decoder_outputs_sample_generator, 1), [hps.batch_size.value, hps.max_dec_sen_num.value, hps.max_dec_steps.value])

        decoder_outputs_given_sample_generator = tf.reshape(decoder_outputs_given_sample_generator, [hps.batch_size.value * hps.max_dec_sen_num.value * hps.max_dec_steps.value, hps.hidden_dim.value])
        decoder_outputs_given_sample_generator = tf.nn.xw_plus_b(decoder_outputs_given_sample_generator, w, v)
        self._sample_given_best_output = tf.reshape(tf.argmax(decoder_outputs_given_sample_generator, 1), [hps.batch_size.value, hps.max_dec_sen_num.value, hps.max_dec_steps.value])

        decoder_outputs_max_generator = tf.reshape(decoder_outputs_max_generator, [hps.batch_size.value * hps.max_dec_sen_num.value * hps.max_dec_steps.value, hps.hidden_dim.value])
        decoder_outputs_max_generator = tf.nn.xw_plus_b(decoder_outputs_max_generator, w, v)
        self._max_best_output = tf.reshape(tf.argmax(decoder_outputs_max_generator, 1), [hps.batch_size.value, hps.max_dec_sen_num.value, hps.max_dec_steps.value])

        loss = tf.contrib.seq2seq.sequence_loss(
            decoder_outputs_pretrain,
            self._target_batch,
            self._dec_padding_mask,
            average_across_timesteps=True,
            average_across_batch=False)

        reward_loss = tf.contrib.seq2seq.sequence_loss(
            decoder_outputs_pretrain,
            self._target_batch,
            self._dec_padding_mask,
            average_across_timesteps=False,
            average_across_batch=False) * self.reward
        reward_loss = tf.reshape(reward_loss, [-1])

        # Update the cost
        self._cost = tf.reduce_mean(loss)
        self._reward_cost = tf.reduce_mean(reward_loss)
        self.optimizer = tf.train.AdagradOptimizer(self._hps.lr.value, initial_accumulator_value=self._hps.adagrad_init_acc.value)

        # Combine LM model loss and AE model loss
        if FLAGS.auto_encoder:
            self._cost += self._AEcost
            self._reward_cost += self._AEreward_cost

  def _add_train_op(self):

    loss_to_minimize = self._cost
    em_tvars = tf.trainable_variables("seq2seq/embedding")
    lm_tvars = tf.trainable_variables("seq2seq/LM_decoder")
    tvars = em_tvars + lm_tvars

    gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

    # Clip the gradients
    grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm.value)

    # Add a summary
    tf.summary.scalar('global_norm', global_norm)

    # Apply adagrad optimizer
    self._train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')

  def _add_reward_train_op(self):

    loss_to_minimize = self._reward_cost
    em_tvars = tf.trainable_variables("seq2seq/embedding")
    lm_tvars = tf.trainable_variables("seq2seq/LM_decoder")
    tvars = em_tvars + lm_tvars

    gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

    # Clip the gradients
    grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm.value)


    self._train_reward_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')

  def _add_AEtrain_op(self):
    if FLAGS.auto_encoder:
      loss_to_minimize = self._AEcost
      em_tvars = tf.trainable_variables("seq2seq/embedding")
      ae_tvars = tf.trainable_variables("seq2seq/AE_decoder")
      tvars = em_tvars + ae_tvars

      gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

      # Clip the gradients
      grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm.value)

      # Add a summary
      tf.summary.scalar('global_norm', global_norm)

      # Apply adagrad optimizer
      self._AEtrain_op = self.AEoptimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')
    else:
      self._AEcost = tf.constant(0)
      self._AEtrain_op = tf.constant(0)

  def _add_AEreward_train_op(self):
    if FLAGS.auto_encoder:
      loss_to_minimize = self._AEreward_cost
      em_tvars = tf.trainable_variables("seq2seq/embedding")
      ae_tvars = tf.trainable_variables("seq2seq/AE_decoder")
      tvars = em_tvars + ae_tvars

      gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

      # Clip the gradients
      grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm.value)


      self._AEtrain_reward_op = self.AEoptimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')
    else:
      self._AEreward_cost = tf.constant(0)
      self._AEtrain_reward_op = tf.constant(0)

  def build_graph(self):

    """Add the placeholders, model, global step, train_op and summaries to the graph"""

    with tf.device("/gpu:"+str(FLAGS.gpuid)):
      tf.logging.info('Building generator graph...')
      t0 = time.time()
      self._add_placeholders()
      self._build_model()
      self.global_step = tf.Variable(0, name='global_step', trainable=False)
      self._add_train_op()
      self._add_reward_train_op()
      self._add_AEtrain_op()
      self._add_AEreward_train_op()
      t1 = time.time()
      tf.logging.info('Time to build graph: %i seconds' % int(t1 - t0))


  def run_pre_train_step(self, sess, batch):
    """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
    feed_dict = self._make_feed_dict(batch)
    to_return = {
        'train_op': self._train_op,
        'loss': self._cost,
        'AEtrain_op': self._AEtrain_op,
        'AEloss': self._AEcost,
        'global_step': self.global_step,
    }
    return sess.run(to_return, feed_dict)


  def run_eval_given_step(self, sess, batch):
      feed_dict = self._make_feed_dict(batch)
      to_return = {
          'generated': self._sample_given_best_output, # (B, S, T)
      }
      return sess.run(to_return, feed_dict)


  def run_train_step(self, sess, batch, reward):
    """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
    feed_dict = self._make_feed_dict(batch)
    feed_dict[self.reward] = reward
    to_return = {
        'train_op': self._train_reward_op,
        'loss': self._reward_cost,
        'AEtrain_op': self._AEtrain_reward_op,
        'AEloss': self._AEreward_cost,
        'global_step': self.global_step,
    }
    return sess.run(to_return, feed_dict)

  def sample_generator(self,sess, batch):
      feed_dict = self._make_feed_dict(batch)
      to_return = {
          'generated': self._sample_best_output,
      }
      return sess.run(to_return, feed_dict)

  def max_generator(self,sess, batch):
      feed_dict = self._make_feed_dict(batch)
      to_return = {
          'generated': self._max_best_output,
      }
      return sess.run(to_return, feed_dict)


