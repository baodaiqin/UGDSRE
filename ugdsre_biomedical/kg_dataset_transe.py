import random
import numpy as np

class KnowledgeGraph:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.n_entity = 0
        self.n_triplet = 0
        self.w_triplet = []
        self.w_triplet_ = []
        self.neg_w_triplet = []
        self.all_e1e2 = []
        self.all_e1 = []
        self.all_e2 = []
        self.load_triplet()
        
    def load_triplet(self):
        fle = open(self.data_dir + 'kg_tuple.npy', 'rb')
        w_triplet = np.load(fle)#e.g., [[e1, e2, r_w1, r_w2, ...], ...]
        fle.close()
        fle_neg = open(self.data_dir + 'kg_tuple_neg.npy', 'rb')
        neg_w_triplet = np.load(fle_neg)
        fle_neg.close()
        self.w_triplet = w_triplet
        self.neg_w_triplet = neg_w_triplet
        self.n_triplet = w_triplet.shape[0]

    def next_pos_batch(self, batch_size):
        rand_idx = np.random.permutation(self.n_triplet)
        start = 0
        end = min(start+batch_size, self.n_triplet)
        pos_batch = np.array(self.w_triplet[rand_idx[start:end],:])
        while start < self.n_triplet:
            yield pos_batch
            start = end
            end = min(start+batch_size, self.n_triplet)
            pos_batch = np.array([self.w_triplet[i,:] for i in rand_idx[start:end]])
            
    def next_neg_batch(self, batch_size):
        rand_idx = np.random.permutation(self.n_triplet)
        start = 0
        end = min(start+batch_size, self.n_triplet)
	neg_batch = np.array([self.neg_w_triplet[i] for i in rand_idx[start:end]])
        while start < self.n_triplet:
            yield neg_batch
            start = end
            end = min(start+batch_size, self.n_triplet)
            neg_batch = np.array([self.neg_w_triplet[i] for i in rand_idx[start:end]])
            
    

        
