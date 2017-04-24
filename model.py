#coding=utf-8
import tensorflow as tf

class GADA:
	def __init__(self, num_units=200, batch_size=32, num_steps=200, num_proj=64,
				learning_rate_g=0.001, learning_rate_d=0.001, learning_rate_c=0.001,
				hidden_size_d=200):
		self.num_units = num_units
		self.batch_size = batch_size
		self.num_steps = num_steps
		self.num_proj = num_proj
		self.hidden_size_d = hidden_size_d
		self.learning_rate_g = learning_rate_g
		self.learning_rate_d = learning_rate_d
		self.learning_rate_c = learning_rate_c
		self.model_name = "GADA"
		
		self.x = tf.placeholder(tf.int32, [batch_size, num_steps])
		self.y_d = tf.placeholder(tf.int32, [batch_size, 2])
		self.y_s = tf.placeholder(tf.int32, [batch_size, 2])

	def generator(self, reuse=False, attention=True):
		if reuse:
			tf.get_variable_scope().reuse_variables()
			
		lstm_cell_f = tf.contrib.rnn.LSTMCell(num_units=self.num_units, num_proj=self.num_proj, reuse=reuse)
		lstm_cell_b = tf.contrib.rnn.LSTMCell(num_units=self.num_units, num_proj=self.num_proj, reuse=reuse)
		state_f = lstm_cell_f.zero_state(self.batch_size, tf.float32)
		state_b = lstm_cell_b.zero_state(self.batch_size, tf.float32)
		state_hist_f = list()
		state_hist_b = list()
		
		for step in xrange(self.num_steps):
			output_f, state_f = lstm_cell_f(self.x[:,step], state_f)
			output_b, state_b = lstm_cell_b(self.x[:,self.num_steps-step-1], state_b)
			
			state_hist_f.append(output_f)
			state_hist_b.append(output_b)
		
		state_hist_b.reverse()
		
		if attention:
			state_hist = tf.concat([state_hist_f, state_hist_b], 1)
			W = tf.get_variable(self.model_name+"_g_W", [2*self.num_proj, 1])
			b = tf.get_variable(self.model_name+"_g_b", [1])
			
			l_att_score = list()
			att_score_sum = tf.constant(0)
			for state in state_hist:
				att_score = tf.math.exp(tf.nn.relu6(tf.matmul(state, W)+b))
				l_att_score.append(att_score)
				att_score_sum += att_score
				
			feature = tf.constant(0, shape=[self.batch_size, 2*self.num_proj])
			for (state,att_score) in zip(state_hist,l_att_score):
				feature += (att_score/att_score_sum*state)
		else:
			feature = tf.concat([output_f,output_b])
			
		return feature
		
# class Generator:
# 	def __init__(self, num_units=200, batch_size=32, num_steps=200):
# 		self.num_units = num_units
# 		self.batch_size = batch_size
# 		self.num_steps = num_steps
# 
# 		self.x = tf.placeholder(tf.int32, [batch_size, num_steps])
# 
# 	def generate(self):
# 
# class Discriminator:
# 	def __init__(self):
# 
# class Classifier:
# 	def __init__(self):
