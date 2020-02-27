from six.moves import xrange

import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest

Linear = core_rnn_cell._Linear

def my_attention_decoder(decoder_inputs,
							initial_state,
							attention_states,
							cell,
							output_size=None,
							num_heads=1,
							loop_function=None,
							dtype=None,
							scope=None,
							initial_state_attention=False,
							aspect_mask=None):
	"""RNN decoder with attention for the sequence-to-sequence model.
	In this context "attention" means that, during decoding, the RNN can look up
	information in the additional tensor attention_states, and it does this by
	focusing on a few entries from the tensor. This model has proven to yield
	especially good results in a number of sequence-to-sequence tasks. This
	implementation is based on http://arxiv.org/abs/1412.7449 (see below for
	details). It is recommended for complex sequence-to-sequence tasks.
	Args:
		decoder_inputs: A list (len=T) of 2D Tensors [(batch_size * sentence_num) x emb_size].
		initial_state: 2D Tensor [batch_size x cell.state_size].
		attention_states: encoder output, 3D Tensor [batch_size x attn_timestep x attn_hidden_dim].
		cell: tf.nn.rnn_cell.RNNCell defining the cell function and size.
		output_size: Size of the output vectors; if None, we use cell.output_size.
		num_heads: Number of attention heads that read from attention_states.
		loop_function: If not None, this function will be applied to i-th output
			in order to generate i+1-th input, and decoder_inputs will be ignored,
			except for the first element ("GO" symbol). This can be used for decoding,
			but also for training to emulate http://arxiv.org/abs/1506.03099.
			Signature -- loop_function(prev, i) = next
				* prev is a 2D Tensor of shape [batch_size x output_size],
				* i is an integer, the step number (when advanced control is needed),
				* next is a 2D Tensor of shape [batch_size x input_size].
		dtype: The dtype to use for the RNN initial state (default: tf.float32).
		scope: VariableScope for the created subgraph; default: "attention_decoder".
		initial_state_attention: If False (default), initial attentions are zero.
			If True, initialize the attentions from the initial state and attention
			states -- useful when we wish to resume decoding from a previously
			stored decoder state and attention states.
	Returns:
		A tuple of the form (outputs, state), where:
			outputs: A list of the same length as decoder_inputs of 2D Tensors of
				shape [batch_size x output_size]. These represent the generated outputs.
				Output i is computed from input i (which is either the i-th element
				of decoder_inputs or loop_function(output {i-1}, i)) as follows.
				First, we run the cell on a combination of the input and previous
				attention masks:
					cell_output, new_state = cell(linear(input, prev_attn), prev_state).
				Then, we calculate new attention masks:
					new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
				and then we calculate the output:
					output = linear(cell_output, new_attn).
			state: The state of each decoder cell the final time-step.
				It is a 2D Tensor of shape [batch_size x cell.state_size].
	Raises:
		ValueError: when num_heads is not positive, there are no inputs, shapes
			of attention_states are not set, or input size cannot be inferred
			from the input.
	"""
	if not decoder_inputs:
		raise ValueError("Must provide at least 1 input to attention decoder.")
	if num_heads < 1:
		raise ValueError("With less than 1 heads, use a non-attention decoder.")
	if attention_states.get_shape()[2].value is None:
		raise ValueError("Shape[2] of attention_states must be known: %s" % attention_states.get_shape())
	if output_size is None:
		output_size = cell.output_size

	with variable_scope.variable_scope(
			scope or "attention_decoder", dtype=dtype) as scope:
		dtype = scope.dtype

		batch_size = array_ops.shape(decoder_inputs[0])[0] # Needed for reshaping.
		attn_timestep = attention_states.get_shape()[1].value # encoder output timestep
		if attn_timestep is None:
			attn_timestep = array_ops.shape(attention_states)[1]
		attn_hidden_dim = attention_states.get_shape()[2].value # encoder output hidden state dimension

		# To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
		hidden = array_ops.reshape(attention_states, [-1, attn_timestep, 1, attn_hidden_dim]) # in CNN input represents: [batch, height, width, channels]
		hidden_features = []
		v = []
		for a in xrange(num_heads):
			k = variable_scope.get_variable("AttnW_%d" % a, [1, 1, attn_hidden_dim, attn_hidden_dim]) # CNN filter
			hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME")) # Do Convolution, get a hidden feature, shape = [batch, height, width, channels]
			v.append(variable_scope.get_variable("AttnV_%d" % a, [attn_hidden_dim]))

		state = initial_state

		def attention(query):
			""" Put attention masks on hidden using hidden_features and query.
				- query: decoder current state, tuple (h, c)
			"""
			ds = [] # Results of attention reads will be stored here.
			if nest.is_sequence(query): # If the query is a tuple, flatten it.
				query_list = nest.flatten(query)
				for q in query_list: # Check that ndims == 2 if specified.
					ndims = q.get_shape().ndims
					if ndims:
						assert ndims == 2
				query = array_ops.concat(query_list, 1)
			for a in xrange(num_heads):
				with variable_scope.variable_scope("Attention_%d" % a):
					y = Linear(query, attn_hidden_dim, True)(query) # transform query (decoder current state) to attn_hidden_dim size
					y = array_ops.reshape(y, [-1, 1, 1, attn_hidden_dim])
					# Attention mask is a softmax of v^T * tanh(...).
					s = math_ops.reduce_sum(v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3]) # hidden feature comes from attention_state, "+" is element-wise plus, v[a] is key
					a = nn_ops.softmax(s)
					# Now calculate the attention-weighted vector d.
					d = math_ops.reduce_sum(array_ops.reshape(a, [-1, attn_timestep, 1, 1]) * hidden, [1, 2])
					ds.append(array_ops.reshape(d, [-1, attn_hidden_dim]))
			return ds

		def aspect_attention(query):
			""" Put attention masks on hidden using hidden_features and query.
				- query: decoder current state, tuple (h, c)
			"""
			ds = [] # Results of attention reads will be stored here.
			if nest.is_sequence(query): # If the query is a tuple, flatten it.
				query_list = nest.flatten(query)
				for q in query_list: # Check that ndims == 2 if specified.
					ndims = q.get_shape().ndims
					if ndims:
						assert ndims == 2
				query = array_ops.concat(query_list, 1)
			for a in xrange(num_heads):
				with variable_scope.variable_scope("Aspect_Attention_%d" % a):
					y = Linear(query, attn_hidden_dim, True)(query) # transform query (decoder current state) to attn_hidden_dim size
					y = array_ops.reshape(y, [-1, 1, 1, attn_hidden_dim])
					# Attention mask is a softmax of v^T * tanh(...).
					s = math_ops.reduce_sum(v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3]) # hidden feature comes from attention_state, "+" is element-wise plus, v[a] is key
					s = s * tf.cast(aspect_mask, tf.float32)
					a = nn_ops.softmax(s)
					# Now calculate the attention-weighted vector d.
					d = math_ops.reduce_sum(array_ops.reshape(a, [-1, attn_timestep, 1, 1]) * hidden, [1, 2])
					ds.append(array_ops.reshape(d, [-1, attn_hidden_dim]))
			return ds

		outputs = []
		prev = None
		attns = [array_ops.zeros([batch_size, attn_hidden_dim], dtype=dtype) for _ in xrange(num_heads)] # list, shape = (B, H) * num_heads
		for a in attns: # Ensure the second shape of attention vectors is set.
			a.set_shape([None, attn_hidden_dim])
		if initial_state_attention:
			attns = attention(initial_state)
		for i, inp in enumerate(decoder_inputs): # Do attention on the same timestep of decoder_input, i = timestep of decoder_inputs, inp = (B*S, E) 
			if i > 0:
				variable_scope.get_variable_scope().reuse_variables()
			# If loop_function is set, we use it instead of decoder_inputs.
			if loop_function is not None and prev is not None:
				with variable_scope.variable_scope("loop_function", reuse=True):
					inp = loop_function(prev, i)
			# Merge input and previous attentions into one vector of the right size.
			emb_size = inp.get_shape().with_rank(2)[1]
			if emb_size.value is None:
				raise ValueError("Could not infer input size from input: %s" % inp.name)

			inputs = [inp] + attns 
			x = Linear(inputs, emb_size, True)(inputs) # transform the last axis to emb_size
			
			# Run the RNN. (decoder cell, x = decoder_input, state = tuple (h, c) decoder previous state)
			cell_output, state = cell(x, state)
			
			# Run the attention mechanism. (state = decoder current state)
			if i == 0 and initial_state_attention:
				with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=True):
					attns = attention(state)
			else:
				attns = attention(state)
			
			# Run the aspect attention mechanism.
			if aspect_mask is not None:
				aspect_attns = aspect_attention(state)
				with variable_scope.variable_scope("Meta_Attention"):
					metas = []
					for a in xrange(num_heads):
						w2 = variable_scope.get_variable("MetaW1_%d" % a, [attn_hidden_dim])
						w3 = variable_scope.get_variable("MetaW2_%d" % a, [attn_hidden_dim])
						meta = math_ops.tanh(w2 * attns[a] + w3 * aspect_attns[a])
						metas.append(meta)
					attns = metas

			with variable_scope.variable_scope("AttnOutputProjection"):
				inputs = [cell_output] + attns
				output = Linear(inputs, output_size, True)(inputs)
			if loop_function is not None:
				prev = output
			outputs.append(output)

	return outputs, state