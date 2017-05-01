#coding=utf-8
import os
import random
import sys

_EMB_SIZE = [25,50,75]
_NUM_UNITS = [50,75,100,200]
_BATCH_SIZE = [16,32]
_NUM_STEPS = [100,200]
_NUM_PROJ = [50,64,75,100,200,300]
_LEARNING_RATE_G = [0.001,0.0001,0.00001,0.000001,0.00000001]
_LEARNING_RATE_D = [0.001,0.0001,0.00001,0.000001,0.00000001]
_LEARNING_RATE_C = [0.001,0.0001,0.00001,0.000001,0.00000001]
_HIDDEN_SIZE_D = [50,75,100,200,300,400]

class config:
    def __init__(self, idx):
        self.save_path = "./save/gada_att_%d" %(idx)
        self.log_path = "./log/config%d.log" %(idx)
        self.emb_size = random.choice(_EMB_SIZE)
        self.souce_dir = "/home/kh/amazon_review/experiment/lem_data/dvd/"
        self.target_dir = "/home/kh/amazon_review/experiment/lem_data/books"
        self.num_units = random.choice(_NUM_UNITS)
	self.is_small = "True"        
        self.batch_size = random.choice(_BATCH_SIZE)
        self.num_steps = self.maxlen = random.choice(_NUM_STEPS)
        self.num_proj = random.choice(_NUM_PROJ)
        self.learning_rate_g = random.choice(_LEARNING_RATE_G)
        self.learning_rate_d = random.choice(_LEARNING_RATE_D)
        self.learning_rate_c = random.choice(_LEARNING_RATE_C)
        self.hidden_size_d = random.choice(_HIDDEN_SIZE_D)
	self.train_spochs = 200        

        self.command = "python run.py --save_path %s \
                            --log_path %s\
                            --emb_size %d\
                            --source_dir %s\
                            --target_dir %s\
                            --num_units %d\
                            --batch_size %d\
                            --num_steps %d\
                            --num_proj %d\
                            --learning_rate_g %g\
                            --learning_rate_d %g\
                            --learning_rate_c %g\
                            --hidden_size_d %d\
                            --is_small %s\
                            --train_epochs %d\
                            --maxlen %d > ./tmp/out%d 2>./tmp/err%d &" %(self.save_path, self.log_path, self.emb_size, self.souce_dir, self.target_dir,
                                            self.num_units, self.batch_size, self.num_steps, self.num_proj,
                                            self.learning_rate_g, self.learning_rate_d, self.learning_rate_c,
                                            self.hidden_size_d, self.is_small, self.train_spochs, self.maxlen, idx, idx)
        
if __name__ == "__main__":
    idx = 0
    l_config = list()
    l_config.append(config(idx))
    idx += 1
    os.system("rm ./tmp/*")
    os.system("rm ./log/*")
    num_idx = int(sys.argv[1])

    while idx < num_idx:
        c = config(idx)
        flag = False
        for t in l_config:
            if t.command == c.command:
                flag = True
                break
        if not flag:
            l_config.append(c)
            idx += 1
            
    for c in l_config:
        os.system(c.command)
