#coding=utf-8
import tensorflow as tf
from reader import get_data, get_batch
from model import GADA

tf.flags.DEFINE_bool("is_train", True, "is_train")
tf.flags.DEFINE_string("save_path", "./save/gada-att", "save_path")
tf.flags.DEFINE_string("emb_path", "/home/kh/amazon_review/experiment/wordvec/all_reviews.txt.dim25", "emb_path")
tf.flags.DEFINE_string("source_dir", None, "source_dir")
tf.flags.DEFINE_string("target_dir", None, "target_dir")
tf.flags.DEFINE_int("num_units", 200, "num_units")
tf.flags.DEFINE_int("batch_size", 32, "batch_size")
tf.flags.DEFINE_int("num_steps", 200, "num_steps")
tf.flags.DEFINE_int("num_proj", 64, "num_proj")
tf.flags.DEFINE_float("learning_rate_g", 0.001, "learning_rate_g")
tf.flags.DEFINE_float("learning_rate_d", 0.001, "learning_rate_d")
tf.flags.DEFINE_float("learning_rate_c", 0.001, "learning_rate_c")
tf.flags.DEFINE_int("hidden_size_d", 200, "hidden_size_d")
tf.flags.DEFINE_float("grad_clip", 5.0, "grad_clip")
tf.flags.DEFINE_int("maxlen", 200, "maxlen")
tf.flags.DEFINE_int("train_epochs", 2000, "train_epochs")

FLAGS = tf.flags

def partial_train(sess, model, dev_model, train_data, dev_data, word_emb):
	#Saver
	saver = tf.train.Saver(tf.trainable_variables())
	
	for slice in xrange(len(train_data)):
		data = train_data[slice]
		for epoch in xrange(FLAGS.train_epochs):
			# train discriminator
			for i in xrange(5):
				_x, _y = get_batch(train_data["all"], FLAGS.batch_size)
				sess.run(model.d_opt, feed_dict={})
			# train generator
			_x, _y = get_batch(train_data["all"], FLAGS.batch_size)
			sess.run(model.g_opt, feed_dict={})
			# train classifier
			for i in xrange(2):
				_x, _y = get_batch(train_data["sentiment"], FLAGS.batch_size)
				sess.run(model.c_opt, feed_dict={})
				
			if epoch%100:
				acc = test(sess, dev_data, dev_model)
				print "slice %d epoch %d: %g" %(slice, epoch, acc)
				
			if epoch%1000 == 0:
				saver.save(sess, FLAGS.save_path)
	
def test(sess, test_data, model=None):
	if not model:
		saver = tf.train.Saver(tf.trainable_variables())
		model = saver.restore(sess, FLAGS.save_path)
	
	true_num = 0
	for term in test_data:
		logits_c = sess.run(model.logits_c, feed_dict={})
		pred_label = 0 if logits_c[0]>logits_c[1] else 1
		if pred_label == term[1]:
			true_num += 1
	return true_num/float(len(test_data))
	
def main(_):
	with tf.Session() as sess:
		train_data, dev_data, test_data, word_emb = get_data(FLAGS.emb_path, FLAGS.source_dir, 
											FLAGS.target_dir, FLAGS.maxlen)
		
		with tf.variable_scope("train", reuse=None):
			model = GADA(word_emb, FLAGS.num_units, FLAGS.batch_size, 
					FLAGS.num_steps, FLAGS.num_proj,
					FLAGS.learning_rate_g, FLAGS.learning_rate_d, FLAGS.learning_rate_c,
					FLAGS.hidden_size_d, FLAGS.grad_clip)
		
		with tf.variable_scope("dev", reuse=True):
			dev_model = GADA(word_emb, FLAGS.num_units, 1, 
					FLAGS.num_steps, FLAGS.num_proj,
					FLAGS.learning_rate_g, FLAGS.learning_rate_d, FLAGS.learning_rate_c,
					FLAGS.hidden_size_d, FLAGS.grad_clip)
		
		with tf.variable_scope("dev", reuse=True):
			test_model = GADA(word_emb, FLAGS.num_units, 1, 
					FLAGS.num_steps, FLAGS.num_proj,
					FLAGS.learning_rate_g, FLAGS.learning_rate_d, FLAGS.learning_rate_c,
					FLAGS.hidden_size_d, FLAGS.grad_clip)
		
		sess.run(tf.initialize_all_variables())
		
		partial_train(sess, model, dev_model, train_data, dev_data, word_emb)
		test(sess, test_model, test_data)

if __name__ == "__main__":
	tf.app.run()