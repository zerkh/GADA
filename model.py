#coding=utf-8
import tensorflow as tf

class GADA:
	def __init__(self, num_units=200, batch_size=32, num_steps=200, learning_rate_g=0.001, learning_rate_d=0.001, learning_rate_c=0.001):
		self.num_units = num_units
		self.batch_size = batch_size
		self.num_steps = num_steps
		self.learning_rate_g = learning_rate_g
		self.learning_rate_d = learning_rate_d
		self.learning_rate_c = learning_rate_c

class Generator:
	def __init__(self, num_units=200, batch_size=32, num_steps=200):
		self.num_units = num_units
		self.batch_size = batch_size
		self.num_steps = num_steps

		self.x = tf.placeholder(tf.int32, [batch_size, num_steps])

	def generate(self):

class Discriminator:
	def __init__(self):

class Classifier:
	def __init__(self):
