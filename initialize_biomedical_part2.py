import numpy as np
import pickle
import random
import sys
from time import time

data_path = './origin_data_type_year/'
export_path_g = './biomedical_part2/'
export_path = './biomedical_part1/'

maxlen = 120
fixlen = 120
kg_fixlen = 20

def pos_embed(x):
    return max(0, min(x + maxlen, maxlen + maxlen + 1))

def init_train_files(name):
    f = open(data_path + name + '.txt','r')
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
	    en1_id, en1_tp = content[0].split('_')
	    en2_id, en2_tp = content[1].split('_')
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
            tup_sent = (tup, sentence, en1_name, en2_name, en1_tp, en2_tp)
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
    sen_head_tying = np.zeros((total), dtype = np.int32)
    sen_tail_tying = np.zeros((total), dtype = np.int32)
    sen_head_type = np.zeros((total), dtype = np.int32)
    sen_tail_type = np.zeros((total), dtype = np.int32)

    for senti, tup_sent in enumerate(lst_tup_sent):
        tup, sent, en1_name, en2_name, en1_tp, en2_tp = tup_sent
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
        sen_head_type[senti] = word2id[en1_tp]
        sen_tail_type[senti] = word2id[en2_tp]

        try:
            en1_id_tying = word2id['%s#ent' % en1_id]
        except KeyError:
            en1_id_tying = word2id[en1_id]

        try:
            en2_id_tying = word2id['%s#ent' % en2_id]
	except KeyError:
            en2_id_tying = word2id[en2_id]

        sen_head_tying[senti] = en1_id_tying
        sen_tail_tying[senti] = en2_id_tying
        
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
            sen_head, sen_tail, sen_head_type, sen_tail_type, sen_head_tying, sen_tail_tying,
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

    return instance_scope_g, sen_word_g, sen_pos1_g, sen_pos2_g

def init_kg():
    f = open(data_path + 'kg/train.txt', 'r')
    all_lw = []
    all_tup = []
    for line in f:
        line = line.strip('\n')
        try:
            e1, e2, rl = line.split('\t')
            tup = (e1, e2, rl)
        except ValueError:
            continue

        lw = [e1, e2] + rl.split('_')
        all_tup.append(tup)
        all_lw.append(lw)
    f.close()

    total = len(all_lw)
    kg_word = np.zeros((total, kg_fixlen), dtype = np.int32)
    kg_tup = np.zeros((total, 3), dtype = np.int32)
    kg_tup = []
    kg_tup_neg = []
    kg_tup_tying = []
    kg_tup_neg_tying = []
    d_tup = {}
    all_e1 = []
    all_e2 = []
    d_tup_tying = {}
    all_e1_tying = []
    all_e2_tying = []
    for s in range(total):
        tup = all_tup[s]
        e1, e2, rl = tup
        try:
            ei1 = word2id[e1]
            ei2 = word2id[e2]
            rli = relation2id[rl]
            tup = [ei1, ei2, rli]
            kg_tup.append(tup)
            d_tup[(ei1, ei2, rli)] = True
            all_e1.append(ei1)
            all_e2.append(ei2)
            try:
                ei1_tying = word2id['%s#ent' % e1]
            except KeyError:
                ei1_tying = ei1
            try:
                ei2_tying = word2id['%s#ent' % e2]
            except KeyError:
                ei2_tying = ei2
            d_tup_tying[(ei1_tying, ei2_tying, rli)] = True
            all_e1_tying.append(ei1_tying)
            all_e2_tying.append(ei2_tying)
            tup_tying = [ei1_tying, ei2_tying, rli]
            kg_tup_tying.append(tup_tying)
        except KeyError:
            pass
        lw = all_lw[s]
        for i in range(kg_fixlen):
            if i < len(lw):
                w = lw[i]
                if w in word2id:
                    kg_word[s][i] = word2id[w]
                else:
                    kg_word[s][i] = word2id['UNK']
            else:
                kg_word[s][i] = word2id['BLANK']

    kg_tup = np.array(kg_tup, dtype=np.int32)
    kg_tup_tying = np.array(kg_tup_tying, dtype=np.int32)
    
    all_e1 = list(set(all_e1))
    all_e2 = list(set(all_e2))
    all_e1_tying = list(set(all_e1_tying))
    all_e2_tying = list(set(all_e2_tying))
    for tup in kg_tup:
            e1 = tup[0]
            e2 = tup[1]
            neg_e1 = e1
            neg_e2 = e2
            ntup = list(tuple(tup))
            while True:
                ht_prob = np.random.binomial(1, 0.5)
                if ht_prob:
                    neg_e1 = random.choice(all_e1)
                else:
                    neg_e2 = random.choice(all_e2)
                ntup[0] = neg_e1
                ntup[1] = neg_e2
                try:
                    d_tup[tuple(ntup)]
                except KeyError:
                    break
            kg_tup_neg.append(ntup)

    for tup in kg_tup_tying:
            e1 = tup[0]
            e2 = tup[1]
            neg_e1 = e1
            neg_e2 = e2
            ntup = list(tuple(tup))
            while True:
                ht_prob = np.random.binomial(1, 0.5)
                if ht_prob:
                    neg_e1 = random.choice(all_e1_tying)
                else:
                    neg_e2 = random.choice(all_e2_tying)
                ntup[0] = neg_e1
                ntup[1] = neg_e2
                try:
                    d_tup_tying[tuple(ntup)]
                except KeyError:
                    break
            kg_tup_neg_tying.append(ntup)

    kg_tup_neg = np.array(kg_tup_neg, dtype=np.int32)
    kg_tup_neg_tying = np.array(kg_tup_neg_tying, dtype=np.int32)
    
    return np.array(kg_word), kg_tup, kg_tup_neg, kg_tup_tying, kg_tup_neg_tying

if __name__ == '__main__':
    start_time = time()
    f_rl2id = open(export_path + 'relation2id.pkl', 'rb')
    relation2id = pickle.load(f_rl2id)
    f_rl2id.close()

    f_word2id = open(data_path + 'word2id.pkl', 'rb')
    word2id = pickle.load(f_word2id)
    f_word2id.close()    

    data_type = 'training'
    print 'Initializing training data ...'
    if data_type == 'training':
        for i in [0]:
            (instance_triple, instance_scope, train_len, train_label, train_word, train_pos1, train_pos2, train_mask,
             train_head, train_tail, train_head_type, train_tail_type, train_head_tying, train_tail_tying,
             train_word_kg, train_pos1_kg, train_pos2_kg, instance_scope_kg,
             train_word_tx, train_pos1_tx, train_pos2_tx, instance_scope_tx,
             train_word_ug, train_pos1_ug, train_pos2_ug, instance_scope_ug)= init_train_files("train%s_sort" % i)
            np.save(export_path_g+'train_word_cross%s' % i, train_word_kg)
            np.save(export_path_g+'train_pos1_cross%s' % i, train_pos1_kg)
            np.save(export_path_g+'train_pos2_cross%s' % i, train_pos2_kg)
            np.save(export_path_g+'train_instance_scope_kg%s' % i, instance_scope_kg)
            np.save(export_path_g+'train_word_cross_tx%s' % i, train_word_tx)
            np.save(export_path_g+'train_pos1_cross_tx%s' % i, train_pos1_tx)
            np.save(export_path_g+'train_pos2_cross_tx%s' % i, train_pos2_tx)
            np.save(export_path_g+'train_instance_scope_tx%s' % i, instance_scope_tx)
            np.save(export_path_g+'train_word_cross_ug%s' % i, train_word_ug)
            np.save(export_path_g+'train_pos1_cross_ug%s' % i, train_pos1_ug)
            np.save(export_path_g+'train_pos2_cross_ug%s' % i, train_pos2_ug)
            np.save(export_path_g+'train_instance_scope_ug%s' % i, instance_scope_ug)

    data_type = 'testing'
    print 'Initializing teting data ...'
    if data_type == 'testing':
        for i in [0]:
            (instance_triple, instance_scope, test_len, test_label, test_word, test_pos1, test_pos2, test_mask,
             test_head, test_tail, test_head_type, test_tail_type, test_head_tying, test_tail_tying,
             test_word_kg, test_pos1_kg, test_pos2_kg, instance_scope_kg,
             test_word_tx, test_pos1_tx, test_pos2_tx, instance_scope_tx,
             test_word_ug, test_pos1_ug, test_pos2_ug, instance_scope_ug)= init_train_files("test%s_sort" % str(i))
            np.save(export_path_g+'test_word_kg%s' % i, test_word_kg)
            np.save(export_path_g+'test_pos1_kg%s' % i, test_pos1_kg)
            np.save(export_path_g+'test_pos2_kg%s' % i, test_pos2_kg)
            np.save(export_path_g+'test_instance_scope_kg%s' % i, instance_scope_kg)
            np.save(export_path_g+'test_word_tx%s' % i, test_word_tx)
            np.save(export_path_g+'test_pos1_tx%s' % i, test_pos1_tx)
            np.save(export_path_g+'test_pos2_tx%s' % i, test_pos2_tx)
            np.save(export_path_g+'test_instance_scope_tx%s' % i, instance_scope_tx)
            np.save(export_path_g+'test_word_ug%s' % i, test_word_ug)
            np.save(export_path_g+'test_pos1_ug%s' % i, test_pos1_ug)
            np.save(export_path_g+'test_pos2_ug%s' % i, test_pos2_ug)
            np.save(export_path_g+'test_instance_scope_ug%s' % i, instance_scope_ug)

    end_time = time()
    time_taken = end_time - start_time
    hours, rest = divmod(time_taken, 3600)
    minutes, seconds = divmod(time_taken, 60)

    print 'Duration: about %s hours %s'	% (hours, rest)
    print 'Duration: about %s minutes %s' % (minutes, seconds)
