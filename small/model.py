#coding=utf-8
import tensorflow as tf
from reader import read_wordvec

class GADA:
	def __init__(self, word_emb, is_training=False, num_units=200, batch_size=32, num_steps=200, num_proj=64,
				learning_rate_g=0.001, learning_rate_d=0.001, learning_rate_c=0.001,
				hidden_size_d=200, grad_clip=5.0):
		self.num_units = num_units
		self.batch_size = batch_size
		self.num_steps = num_steps
		self.num_proj = num_proj
		self.hidden_size_d = hidden_size_d
		self.learning_rate_g = learning_rate_g
		self.learning_rate_d = learning_rate_d
		self.learning_rate_c = learning_rate_c
		self.grad_clip = grad_clip
		self.model_name = "GADA"
		
		vocab_size = len(word_emb)
		size = len(word_emb[0])
		self.embedding = tf.Variable(word_emb, name="embedding")

		self.x = tf.placeholder(tf.int32, [batch_size, num_steps])
		self.y_d = tf.placeholder(tf.int32, [batch_size, 2])
		self.y_s = tf.placeholder(tf.int32, [batch_size, 2])
		self.x_vec = tf.nn.embedding_lookup(self.embedding, self.x)

		self.feature = self.generator()
		self.logits_d = self.discriminator(self.feature)
		self.logits_c = self.classifier(self.feature)

		self.create_loss_terms()

		if is_training:
			# get variables
			self.all_vars = tf.trainable_variables()
	
			self.g_vars = [var for var in self.all_vars if (self.model_name+'_g_') in var.name]
			self.d_vars = [var for var in self.all_vars if (self.model_name+'_d_') in var.name]
			self.c_vars = [var for var in self.all_vars if (self.model_name+'_c_') in var.name]
	
			self.gc_vars = self.g_vars+self.c_vars+[self.embedding]
			self.g_vars += [self.embedding]
	
			# get variables' grad
			g_grads,_ = tf.clip_by_global_norm(tf.gradients(self.loss_g, self.g_vars), self.grad_clip)
			d_grads,_ = tf.clip_by_global_norm(tf.gradients(self.loss_d, self.d_vars), self.grad_clip)
			c_grads,_ = tf.clip_by_global_norm(tf.gradients(self.loss_c, self.c_vars), self.grad_clip)
	
			g_optimizer = tf.train.AdamOptimizer(self.learning_rate_g)
			d_optimizer = tf.train.AdamOptimizer(self.learning_rate_d)
			c_optimizer = tf.train.AdamOptimizer(self.learning_rate_c)
	
			self.g_opt = g_optimizer.apply_gradients(zip(g_grads, self.g_vars))
			self.d_opt = d_optimizer.apply_gradients(zip(d_grads, self.d_vars))
			self.c_opt = c_optimizer.apply_gradients(zip(c_grads, self.c_vars))


	def generator(self, reuse=False, attention=True):
		if reuse:
			tf.get_variable_scope().reuse_variables()
			
		with tf.variable_scope(self.model_name+"_g_lstm_f"):
			lstm_cell_f = tf.contrib.rnn.LSTMCell(num_units=self.num_units, num_proj=self.num_proj)
		with tf.variable_scope(self.model_name+"_g_lstm_b"):
			lstm_cell_b = tf.contrib.rnn.LSTMCell(num_units=self.num_units, num_proj=self.num_proj)
		state_f = lstm_cell_f.zero_state(self.batch_size, tf.float32)
		state_b = lstm_cell_b.zero_state(self.batch_size, tf.float32)
		state_hist_f = list()
		state_hist_b = list()
		
		for step in xrange(self.num_steps):
			with tf.variable_scope(self.model_name+"_g_lstm_f"):
				if step > 0: tf.get_variable_scope().reuse_variables()
				output_f, state_f = lstm_cell_f(self.x_vec[:,step,:], state_f)
			with tf.variable_scope(self.model_name+"_g_lstm_b"):
				if step > 0: tf.get_variable_scope().reuse_variables()
				output_b, state_b = lstm_cell_b(self.x_vec[:,self.num_steps-step-1,:], state_b)
			
			state_hist_f.append(output_f)
			state_hist_b.append(output_b)
		
		state_hist_b.reverse()
		
		if attention:
			state_hist = tf.concat([state_hist_f, state_hist_b], 2)
			W = tf.get_variable(self.model_name+"_g_W", [2*self.num_proj, 1])
			b = tf.get_variable(self.model_name+"_g_b", [1])
			
			l_att_score = list()
			att_score_sum = tf.constant(0.0)
			for step in xrange(self.num_steps):
				state = state_hist[step,:,:]
				att_score = tf.exp(tf.nn.relu6(tf.matmul(state, W)+b))
				l_att_score.append(att_score)
				att_score_sum += att_score
				
			feature = tf.constant(0.0, shape=[self.batch_size, 2*self.num_proj])
			for step in xrange(self.num_steps):
				state = state_hist[step,:,:]
				att_score = l_att_score[step]
				feature += (att_score/att_score_sum*state)
		else:
			feature = tf.concat([output_f,output_b])
			
		return feature

	def discriminator(self, feature, reuse=False):
		if reuse:
			tf.get_varibale_scope().reuse_variables()

		W1 = tf.get_variable(self.model_name+"_d_W1", [2*self.num_proj, self.hidden_size_d])
		b1 = tf.get_variable(self.model_name+"_d_b1", [1, self.hidden_size_d])

		W2 = tf.get_variable(self.model_name+"_d_W2", [self.hidden_size_d, self.hidden_size_d])
		b2 = tf.get_variable(self.model_name+"_d_b2", [1, self.hidden_size_d])

		softmax_w = tf.get_variable(self.model_name+"_d_softmax_w", [self.hidden_size_d, 2])
		softmax_b = tf.get_variable(self.model_name+"_d_softmax_b", [1, 2])

		layer1 = tf.nn.relu(tf.matmul(feature,W1)+b1)
		layer2 = tf.nn.relu(tf.matmul(layer1,W2)+b2)
		logits = tf.matmul(layer2, softmax_w)+softmax_b

		return logits

	def classifier(self, feature, reuse=False):
		if reuse:
			tf.get_varibale_scope().reuse_variables()

		W1 = tf.get_variable(self.model_name+"_c_W1", [2*self.num_proj, self.hidden_size_d])
		b1 = tf.get_variable(self.model_name+"_c_b1", [1, self.hidden_size_d])

		W2 = tf.get_variable(self.model_name+"_c_W2", [self.hidden_size_d, self.hidden_size_d])
		b2 = tf.get_variable(self.model_name+"_c_b2", [1, self.hidden_size_d])

		softmax_w = tf.get_variable(self.model_name+"_c_softmax_w", [self.hidden_size_d, 2])
		softmax_b = tf.get_variable(self.model_name+"_c_softmax_b", [1, 2])

		layer1 = tf.nn.relu(tf.matmul(feature,W1)+b1)
		layer2 = tf.nn.relu(tf.matmul(layer1,W2)+b2)
		logits = tf.matmul(layer2, softmax_w)+softmax_b

		return logits

	def create_loss_terms(self):
		self.loss_g = -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_d, logits=self.logits_d))
		self.loss_d = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_d, logits=self.logits_d))
		self.loss_c = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_s, logits=self.logits_c))

if __name__ == "__main__":
	d_word_idx, d_idx_word, word_emb = read_wordvec("/home/kh/amazon_review/experiment/wordvec/all_reviews.txt.dim25")
	m = GADA(word_emb)
#	feat = m.generator()
#	logits = m.discriminator(feat)
