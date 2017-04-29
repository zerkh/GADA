#coding=utf-8
import random

def read_wordvec(path):
	fin = open(path)
	lines = fin.readlines()

	vocb_size, vec_dim = int(lines[0].split(" ")[0].strip()), int(lines[0].split(" ")[1].strip())
	d_word_idx = dict()
	d_idx_word = dict()
	l_wordvec = list()

	# add special tokens
	d_word_idx["<UNK>"] = 0
	d_idx_word[0] = "<UNK>"
	l_wordvec.append([random.random() for _ in xrange(vec_dim)])
	
	d_word_idx["<PAD>"] = 1
	d_idx_word[1] = "<PAD>"
	l_wordvec.append([0.0]*vec_dim)

	for idx in range(1, len(lines)):
		parts = lines[idx].strip().split(" ")
		word = parts[0]
		vec = list()
		for d in range(1, len(parts)):
			vec.append(float(parts[d]))
		d_word_idx[word] = idx+1
		d_idx_word[idx+1] = word
		l_wordvec.append(vec)

	return d_word_idx, d_idx_word, l_wordvec

def read_sentence(path, d_word_idx, d_idx_word, maxlen=200):
	fin = open(path)
	UNK_idx = d_word_idx["<UNK>"]
	PAD_idx = d_word_idx["<PAD>"]

	l_sentence = list()
	for line in fin:
		l_words = line.strip().split(" ")
		l_idxs = list()
		for word in l_words:
			if d_word_idx.has_key(word):
				l_idxs.append(d_word_idx[word])
			else:
				l_idxs.append(UNK_idx)

		l_sentence.append(l_idxs)

	for i in xrange(len(l_sentence)):
		if len(l_sentence[i]) > maxlen:
			l_sentence[i] = l_sentence[i][:maxlen]
		for _ in range(len(l_sentence[i]), maxlen):
			l_sentence[i].append(PAD_idx)

	return l_sentence

def get_sentence_len(sent):
	index = 0
	while sent[index] != 1 and index < len(sent):
		index += 1
	
	return index

def get_data(emb_path, source_dir, target_dir, maxlen):
	d_word_idx, d_idx_word, word_emb = read_wordvec(emb_path)
	src_all_sent = read_sentence(source_dir+"all.review", d_word_idx, d_idx_word, maxlen)
	src_pos_sent = read_sentence(source_dir+"positive.review", d_word_idx, d_idx_word, maxlen)
	src_neg_sent = read_sentence(source_dir+"negative.review", d_word_idx, d_idx_word, maxlen)
	tar_all_sent = read_sentence(source_dir+"all.review", d_word_idx, d_idx_word, maxlen)
	tar_pos_sent = read_sentence(source_dir+"positive.review", d_word_idx, d_idx_word, maxlen)
	tar_neg_sent = read_sentence(source_dir+"negative.review", d_word_idx, d_idx_word, maxlen)
	
	train_data = list()
	# training data
	if len(src_all_sent) < len(tar_all_sent):
		all_sent = [(sent,0) for sent in src_all_sent]
		all_sent += [(sent,1) for sent in tar_all_sent[:len(src_all_sent)]]
	else:
		all_sent = [(sent,0) for sent in src_all_sent[:len(tar_all_sent)]]
		all_sent += [(sent,1) for sent in tar_all_sent]
		
	if len(src_pos_sent) < len(src_neg_sent):
		senti_sent = [(sent,0) for sent in src_pos_sent]
		senti_sent += [(sent,1) for sent in src_neg_sent[:len(src_pos_sent)]]
	else:
		senti_sent = [(sent,0) for sent in src_pos_sent[:len(src_neg_sent)]]
		senti_sent += [(sent,1) for sent in src_neg_sent]
		
	all_sent = sorted(all_sent, key=lambda x:get_sentence_len(x[0]))
	senti_sent = sorted(senti_sent, key=lambda x:get_sentence_len(x[0]))
	
	num_slides_all = len(all_sent)/3
	num_slides_senti = len(senti_sent)/3
	for i in xrange(3):
		p_all_sent = all_sent[i*num_slides_all,(i+1)*num_slides_all]
		p_senti_sent = senti_sent[i*num_slides_senti,(i+1)*num_slides_senti]
		
		p_train_data = dict()
		p_train_data["all"] = p_all_sent
		p_train_data["sentiment"] = p_senti_sent
		train_data.append(p_train_data)
	
	# testing data
	min_len = len(tar_pos_sent) if len(tar_pos_sent)<len(tar_neg_sent) else len(tar_neg_sent)
	test_pos_data = [(sent,0) for sent in tar_pos_sent[:min_len]]
	test_neg_data = [(sent,1) for sent in tar_neg_sent[:min_len]]
	
	test_data = test_pos_data[:0.9*min_len]
	test_data += test_neg_data[:0.9*min_len]
	
	dev_data = test_pos_data[0.9*min_len:]
	dev_data += test_neg_data[0.9*min_len:]
	
	return train_data, dev_data, test_data

def get_batch(data, batch_size):
	feat = list()
	target = list()
	
	p_data = random.sample(data, batch_size)
	
	for term in p_data:
		feat.append(term[0])
		target.append(term[1])
		
	return feat, target

if __name__ == "__main__":
	d_word_idx, d_idx_word, l_wordvec = read_wordvec("/home/kh/amazon_review/experiment/wordvec/all_reviews.txt.dim25")
	l_sentence = read_sentence("/home/kh/amazon_review/experiment/lem_data/apparel/all.review", d_word_idx, d_idx_word)
