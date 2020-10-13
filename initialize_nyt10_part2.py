import numpy as np
import pickle
import random

data_path = './origin_data_nyt10/'
export_path_g = './nyt10_part2/'
export_path = './nyt10_part1/'

maxlen = 120
fixlen = 120
kg_fixlen = 20

def pos_embed(x):
    return max(0, min(x + maxlen, maxlen + maxlen + 1))

def init_train_files(name):
    f = open(data_path + 'text/' + name + '.txt','r')
    instance_scope = []
    instance_scope_g = []
    instance_triple = []
    instance_triple_g = []

    d_tup_lsent = {}
    total = 0
    lst_tup_sent = []
    total_path = 0
    d_tup_lpath = {}

    lst_tup_path_kg = []
    lst_tup_path_tx = []
    lst_tup_path_ug = []
    d_tup_lpath_kg = {}
    d_tup_lpath_tx = {}
    d_tup_lpath_ug = {}
    
    for line in f:
        line = line.strip('\n')
        content = line.split('\t')
        try:
	    sentence = content[5]
            sentence_cross = content[6]
            sentence_cross3 = content[7]
            sentence_cross4 = content[8]
	    en1_id = content[0]
	    en2_id = content[1]
	    en1_name = content[2]
	    en2_name = content[3]
	    rel_name = content[4]
	    if rel_name in relation2id:
	        relation = relation2id[rel_name]
	    else:
	        continue
        except IndexError:
            continue

        tup = (en1_id,en2_id,relation)
        
        if 'PADDING' not in sentence:
            tup_sent = (tup, sentence, en1_name, en2_name)
            lst_tup_sent.append(tup_sent)
            d_tup_lsent.setdefault(tup, []).append(sentence)
            total += 1

        if 'PADDING' not in sentence_cross:
            tup_path = (tup, sentence_cross)
            lst_tup_path_kg.append(tup_path)
            d_tup_lpath_kg.setdefault(tup, []).append(sentence_cross)
            total_path += 1
        if 'PADDING' not in sentence_cross3:
            tup_path = (tup, sentence_cross3)
            lst_tup_path_tx.append(tup_path)
            d_tup_lpath_tx.setdefault(tup, []).append(sentence_cross3)
            total_path += 1
        if 'PADDING' not in sentence_cross4:
            tup_path = (tup, sentence_cross4)
            lst_tup_path_ug.append(tup_path)
            d_tup_lpath_ug.setdefault(tup, []).append(sentence_cross4)
            total_path += 1

        try:
            d_tup_lpath_kg[tup]
        except KeyError:
            tup_path = (tup, 'PADDING')
            d_tup_lpath_kg.setdefault(tup, []).append(tup_path)
            lst_tup_path_kg.append(tup_path)

        try:
            d_tup_lpath_tx[tup]
        except KeyError:
            tup_path = (tup, 'PADDING')
            d_tup_lpath_tx.setdefault(tup, []).append(tup_path)
            lst_tup_path_tx.append(tup_path)

        try:
            d_tup_lpath_ug[tup]
        except KeyError:
            tup_path = (tup, 'PADDING')
            d_tup_lpath_ug.setdefault(tup, []).append(tup_path)
            lst_tup_path_ug.append(tup_path)
            
    f.close()
    
    sen_word = np.zeros((total, fixlen), dtype = np.int32)
    sen_pos1 = np.zeros((total, fixlen), dtype = np.int32)
    sen_pos2 = np.zeros((total, fixlen), dtype = np.int32)

    sen_mask = np.zeros((total, fixlen), dtype = np.int32)
    sen_len = np.zeros((total), dtype = np.int32)
    sen_label = np.zeros((total), dtype = np.int32)
    sen_head = np.zeros((total), dtype = np.int32)
    sen_tail = np.zeros((total), dtype = np.int32)
    
    for senti, tup_sent in enumerate(lst_tup_sent):
        tup, sent, en1_name, en2_name = tup_sent
        en1_id, en2_id, relation = tup
        sentence = sent.split()
	if instance_triple == [] or instance_triple[len(instance_triple) - 1] != tup:
	    instance_triple.append(tup)
	    instance_scope.append([senti,senti])
	instance_scope[len(instance_triple) - 1][1] = senti
        
	if (senti+1) % 100 == 0:
            sys.stdout.flush()
            pr_out = 'initializing sentence evidences ... %0.3f' % ((1.0 + senti)/total)
            sys.stdout.write(pr_out+'\r')
            sys.stdout.flush()
            
        sen_head[senti] = word2id[en1_id]
        sen_tail[senti] = word2id[en2_id]
        
	en1pos = 0
	en2pos = 0
	for i in range(len(sentence)):
	    if sentence[i] == en1_name:
		en1pos = i
	    if sentence[i] == en2_name:
		en2pos = i
                
	en_first = min(en1pos,en2pos)
	en_second = en1pos + en2pos - en_first
	for i in range(fixlen):
	    sen_word[senti][i] = word2id['BLANK']            
	    sen_pos1[senti][i] = pos_embed(i - en1pos)
	    sen_pos2[senti][i] = pos_embed(i - en2pos)

	    if i >= len(sentence):
		sen_mask[senti][i] = 0
	    elif i - en_first<=0:
		sen_mask[senti][i] = 1
	    elif i - en_second<=0:
		sen_mask[senti][i] = 2
	    else:
		sen_mask[senti][i] = 3
	for i, word in enumerate(sentence):
	    if i >= fixlen:
		break
	    elif not word in word2id:
		sen_word[senti][i] = word2id['UNK']
	    else:
		sen_word[senti][i] = word2id[word]

	sen_len[senti] = min(fixlen, len(sentence))
	sen_label[senti] = relation

    sys.stdout.write('\n')
    kg_scope, path_kg_w, path_kg_pos1, path_kg_pos2 = initiate_path(lst_tup_path_kg, '1')
    tx_scope, path_tx_w, path_tx_pos1, path_tx_pos2 = initiate_path(lst_tup_path_tx, '2')
    ug_scope, path_ug_w, path_ug_pos1, path_ug_pos2 = initiate_path(lst_tup_path_ug, '3')

    assert len(instance_scope) == len(kg_scope) == len(tx_scope) == len(ug_scope)
    
    return (np.array(instance_triple), np.array(instance_scope), sen_len, sen_label, sen_word, sen_pos1, sen_pos2, sen_mask,
            sen_head, sen_tail,
            path_kg_w, path_kg_pos1, path_kg_pos2, np.array(kg_scope),
            path_tx_w, path_tx_pos1, path_tx_pos2, np.array(tx_scope),
            path_ug_w, path_ug_pos1, path_ug_pos2, np.array(ug_scope))

def initiate_path(lst_tup_path, parti):        
    instance_triple_g = []
    instance_scope_g = []
    total_path = len(lst_tup_path)

    sen_word_g = np.zeros((total_path, fixlen), dtype = np.int32)
    sen_pos1_g = np.zeros((total_path, fixlen), dtype = np.int32)
    sen_pos2_g = np.zeros((total_path, fixlen), dtype = np.int32)
    
    for pathi, tup_path in enumerate(lst_tup_path):
        tup, path = tup_path
        if instance_triple_g == [] or instance_triple_g[len(instance_triple_g) - 1] != tup:
            instance_triple_g.append(tup)
            instance_scope_g.append([pathi,pathi])
        instance_scope_g[len(instance_triple_g) - 1][1] = pathi
        if (pathi+1) % 100 == 0:
            sys.stdout.flush()
            pr_out = 'initializing path evidences %s/3 ... %0.3f' % (parti, ((1.0 + pathi)/total_path))
            sys.stdout.write(pr_out+'\r')
            sys.stdout.flush()
            
        sentence_cross = path.split()
        lst_e1 = [i for i, w in enumerate(sentence_cross) if w == '<e1>']
        lst_e2 = [i for i, w in enumerate(sentence_cross) if w == '<e2>']
        e1i = 0
        e2i = 0
        for i, word in enumerate(sentence_cross):
            if i+1 >= fixlen:
                break
            sen_word_g[pathi][i] = word2id['BLANK']
            if word in ['</s1>', '</s2>']:
                e1i += 1
                e2i += 1

            try:
                en1pos_cross = lst_e1[e1i]
                en2pos_cross = lst_e2[e2i]
            except IndexError:
                en1pos_cross = 0
                en2pos_cross = len(sentence_cross) + 1
                
            if i >= fixlen:
                break
            if not word in word2id:
                sen_word_g[pathi][i] = word2id['UNK']
            else:
                sen_word_g[pathi][i] = word2id[word]

            sen_pos1_g[pathi][i] = pos_embed(i - en1pos_cross)
            sen_pos2_g[pathi][i] = pos_embed(i - en2pos_cross)
    sys.stdout.write('\n')
    
    return instance_scope_g, sen_word_g, sen_pos1_g, sen_pos2_g

if __name__ == '__main__':
    import sys
    f_rl2id = open(export_path + 'relation2id.pkl', 'rb')
    relation2id = pickle.load(f_rl2id)
    f_rl2id.close()
    
    f_word2id = open(export_path + 'word2id.pkl', 'rb')
    word2id = pickle.load(f_word2id)
    f_word2id.close()    

    data_type = 'training'
    print 'Initializing training data ...'
    if data_type == 'training':
        (instance_triple, instance_scope, train_len, train_label, train_word, train_pos1, train_pos2, train_mask,
         train_head, train_tail,
         train_word_kg, train_pos1_kg, train_pos2_kg, instance_scope_kg,
         train_word_tx, train_pos1_tx, train_pos2_tx, instance_scope_tx,
         train_word_ug, train_pos1_ug, train_pos2_ug, instance_scope_ug)= init_train_files("train")
        np.save(export_path_g+'train_word_cross_kg', train_word_kg)
        np.save(export_path_g+'train_pos1_cross_kg', train_pos1_kg)
        np.save(export_path_g+'train_pos2_cross_kg', train_pos2_kg)
        np.save(export_path_g+'train_instance_scope_kg', instance_scope_kg)
        np.save(export_path_g+'train_word_cross_tx', train_word_tx)
        np.save(export_path_g+'train_pos1_cross_tx', train_pos1_tx)
        np.save(export_path_g+'train_pos2_cross_tx', train_pos2_tx)
        np.save(export_path_g+'train_instance_scope_tx', instance_scope_tx)
        np.save(export_path_g+'train_word_cross_ug', train_word_ug)
        np.save(export_path_g+'train_pos1_cross_ug', train_pos1_ug)
        np.save(export_path_g+'train_pos2_cross_ug', train_pos2_ug)
        np.save(export_path_g+'train_instance_scope_ug', instance_scope_ug)
    
    data_type = 'testing'
    print 'Initializing teting data ...'
    if data_type == 'testing':
        (instance_triple, instance_scope, test_len, test_label, test_word, test_pos1, test_pos2, test_mask,
         test_head, test_tail,
         test_word_kg, test_pos1_kg, test_pos2_kg, instance_scope_kg,
         test_word_tx, test_pos1_tx, test_pos2_tx, instance_scope_tx,
         test_word_ug, test_pos1_ug, test_pos2_ug, instance_scope_ug)= init_train_files("test")
        np.save(export_path_g+'test_word_kg', test_word_kg)
        np.save(export_path_g+'test_pos1_kg', test_pos1_kg)
        np.save(export_path_g+'test_pos2_kg', test_pos2_kg)
        np.save(export_path_g+'test_instance_scope_kg', instance_scope_kg)
        np.save(export_path_g+'test_word_tx', test_word_tx)
        np.save(export_path_g+'test_pos1_tx', test_pos1_tx)
        np.save(export_path_g+'test_pos2_tx', test_pos2_tx)
        np.save(export_path_g+'test_instance_scope_tx', instance_scope_tx)
        np.save(export_path_g+'test_word_ug', test_word_ug)
        np.save(export_path_g+'test_pos1_ug', test_pos1_ug)
        np.save(export_path_g+'test_pos2_ug', test_pos2_ug)
        np.save(export_path_g+'test_instance_scope_ug', instance_scope_ug)
        
