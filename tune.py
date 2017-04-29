#coding=utf-8
import os
import random

tf.flags.DEFINE_string("save_path", "./save/gada-att", "save_path")
tf.flags.DEFINE_string("log_path", "./log/config.log", "save_path")
tf.flags.DEFINE_integer("emb_size", 25, "num_units")
tf.flags.DEFINE_string("source_dir", None, "source_dir")
tf.flags.DEFINE_string("target_dir", None, "target_dir")
tf.flags.DEFINE_integer("num_units", 200, "num_units")
tf.flags.DEFINE_integer("batch_size", 32, "batch_size")
tf.flags.DEFINE_integer("num_steps", 200, "num_steps")
tf.flags.DEFINE_integer("num_proj", 64, "num_proj")
tf.flags.DEFINE_float("learning_rate_g", 0.001, "learning_rate_g")
tf.flags.DEFINE_float("learning_rate_d", 0.001, "learning_rate_d")
tf.flags.DEFINE_float("learning_rate_c", 0.001, "learning_rate_c")
tf.flags.DEFINE_integer("hidden_size_d", 200, "hidden_size_d")
tf.flags.DEFINE_float("grad_clip", 5.0, "grad_clip")
tf.flags.DEFINE_integer("maxlen", 200, "maxlen")
tf.flags.DEFINE_integer("train_epochs", 1, "train_epochs")

_EMB_SIZE = [25,50,75,100]
_NUM_UNITS = [50,75,100,200]
_BATCH_SIZE = [16,32,64,128]
_NUM_STEPS = [100,200,300]
_NUM_PROJ = [50,64,75,100,200,300]
_LEARNING_RATE_G = [0.001,0.0001,0.00001,0.000001,0.00000001]
_LEARNING_RATE_D = [0.001,0.0001,0.00001,0.000001,0.00000001]
_LEARNING_RATE_C = [0.001,0.0001,0.00001,0.000001,0.00000001]
_HIDDEN_SIZE_D = [50,75,100,200,300,400]

class config:
    def __init__(self, idx):
        self.save_path = "./save/gada_att_%d" %(idx)
        self.log_path = "./log/config%d.log" %(idx)
        self.emb_size = 
        
if __name__ == "__main__":
    