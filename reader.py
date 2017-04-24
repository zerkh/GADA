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

def read_sentence(path, d_word_idx, d_idx_word, maxlen=100):
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

		if len(l_idxs)>maxlen:
			maxlen = len(l_idxs)
		l_sentence.append(l_idxs)

	for i in xrange(len(l_sentence)):
		for _ in range(len(l_sentence[i]), maxlen):
			l_sentence[i].append(PAD_idx)

	return l_sentence

if __name__ == "__main__":
	d_word_idx, d_idx_word, l_wordvec = read_wordvec("/home/kh/amazon_review/experiment/wordvec/all_reviews.txt.dim25")
	read_sentence("/home/kh/amazon_review/experiment/lem_data/apparel/all.review", d_word_idx, d_idx_word)
