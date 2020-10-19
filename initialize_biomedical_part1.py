import numpy as np
import pickle
import random
import sys
from time import time

data_path = './origin_data_type_year/'
export_path = '/home/dq/biomedical_part1/'

maxlen = 120
fixlen = 120
kg_fixlen = 20

def pos_embed(x):
    return max(0, min(x + maxlen, maxlen + maxlen + 1))

def init_word(word2id, sfout_vec):
    all_w = []
    vec = load_w2v() 

    nb_vec, dim_vec = vec.shape
    nvec = np.ones((len(word2id)-nb_vec, dim_vec), dtype = np.float32)
    new_vec = np.concatenate((vec, nvec), axis=0)
    fout_vec = open(sfout_vec, 'wb')
    np.save(fout_vec, new_vec)
    fout_vec.close()

def load_w2v():
    word2ind = {}
    f = open(data_path + 'text/vec.txt', "r")
    f_ent = open(data_path + 'text/vec_ent.txt', "r")
    total, size = f.readline().strip().split()[:2]
    total_ent, size_ent = f_ent.readline().strip().split()[:2]
    total = (int)(total)
    word_size = (int)(size)
    total_ent = int(total_ent)
    vec = np.ones((total + total_ent, word_size), dtype = np.float32)
    for i in range(total):
        content = f.readline().strip().split()
        w = content[0]
        if w not in word2ind:
            wi = len(word2ind)
            word2ind[w] = wi
        wi = word2ind[w]
        for j in range(word_size):
            vec[wi][j] = (float)(content[j+1])
    f.close()

    for i in range(total_ent):
        content = f_ent.readline().strip().split()
        w = content[0]
        if w not in word2ind:
            wi = len(word2ind)
            word2ind[w] = wi
        wi = word2ind[w]
        for j in range(word_size):
            vec[wi][j] = (float)(content[j+1])
    f_ent.close()

    return word2ind, vec
    
def init_relation(sfout):
    relation2id = {}
    with open(data_path + 'text/relation2id.txt', 'rb') as fle:
        for line in fle:
            line = line.strip('\n')
            try:
                rl, rli = line.split()
            except ValueError:
                continue
            relation2id[rl] = int(rli)

    fout = open(sfout, 'wb')
    fout = open(sfout, 'wb')
    pickle.dump(relation2id, fout)
    fout.close()
    return relation2id
    
def sort_files(name, nb_gro):
    d_ep_na = {}
    d_ep_nona = {}
    hash = {}
    f = open(data_path + "text/" + name + '.txt','r')
    s = 0
    while True:
	content = f.readline()
	if content == '':
	    break
	origin_data = content
	content = content.strip().split()
	en1_id = content[0]
	en2_id = content[1]
	rel_name = content[4]
        ep = (en1_id, en2_id)
        if rel_name != 'NA':
            d_ep_nona[ep] = True
        else:
            d_ep_na[ep] = True

    f.close()

    all_ep_na = d_ep_na.keys()
    all_ep_nona = d_ep_nona.keys()

    all_ep = all_ep_nona + all_ep_na
    target_ep = {ep:True for ep in all_ep}
    f = open(data_path + "text/" + name + '.txt','r')
    while True:
        content = f.readline()
        if content == '':
            break
        s = s + 1
        origin_data = content
        content = content.strip().split()
        en1_id = content[0]
        en2_id = content[1]
        rel_name = content[4]
        ep = (en1_id, en2_id)
        try:
            target_ep[ep]
        except KeyError:
            continue
        
	if rel_name in relation2id:
	    relation = relation2id[rel_name]
	else:
	    break
	id1 = str(en1_id)+"#"+str(en2_id)
	id2 = str(relation)
	if not id1 in hash:
	    hash[id1] = {}
	if not id2 in hash[id1]:
	    hash[id1][id2] = []
	hash[id1][id2].append(origin_data)
    f.close()

    if nb_gro > 1:
        all_ep = hash.keys()
        nb_ep = len(all_ep)
        sz_gro = int(nb_ep/(nb_gro+0.0))
        lst_gro = []
        for i in range(0, nb_ep, sz_gro):
            gro = all_ep[i:i+sz_gro]
            lst_gro.append(gro)

        for i, gro in enumerate(lst_gro):
            lst_lout = []
            f = open(data_path + name + "%s_sort.txt" % i, "w")
            
            for i in gro:
                for j in hash[i]:
                    for k in hash[i][j]:
                        lst_lout.append(k)
            f.write("%d\n"%len(lst_lout))
            for lout in lst_lout:
                f.write(lout)
            f.close()
        
    else:
        f = open(data_path + name + "_sort.txt", "w")
        f.write("%d\n"%(s))
        for i in hash:
	    for j in hash[i]:
	        for k in hash[i][j]:
		    f.write(k)
        f.close()

def init_train_files(name):
    f = open(data_path + name + '.txt','r')
    instance_scope = []
    instance_scope_g = []
    instance_triple = []
    instance_triple_g = []

    d_tup_lsent = {}
    total = 0
    lst_tup_path = []
    lst_tup_sent = []
    total_path = 0
    d_tup_lpath = {}
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
            lst_tup_path.append(tup_path)
            d_tup_lpath.setdefault(tup, []).append(sentence_cross)
            total_path += 1
        if 'PADDING' not in sentence_cross3:
            tup_path = (tup, sentence_cross3)
            lst_tup_path.append(tup_path)
            d_tup_lpath.setdefault(tup, []).append(sentence_cross3)
            total_path += 1
        if 'PADDING' not in sentence_cross4:
            tup_path = (tup, sentence_cross4)
            lst_tup_path.append(tup_path)
            d_tup_lpath.setdefault(tup, []).append(sentence_cross4)
            total_path += 1

        try:
            d_tup_lpath[tup]
        except KeyError:
            tup_path = (tup, 'PADDING')
            d_tup_lpath.setdefault(tup, []).append(tup_path)
            lst_tup_path.append(tup_path)
            total_path += 1
            
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
    sen_word_cross = np.zeros((total_path, fixlen), dtype = np.int32)
    sen_pos1_cross = np.zeros((total_path, fixlen), dtype = np.int32)
    sen_pos2_cross = np.zeros((total_path, fixlen), dtype = np.int32)
    
    for pathi, tup_path in enumerate(lst_tup_path):
        tup, path = tup_path
        if instance_triple_g == [] or instance_triple_g[len(instance_triple_g) - 1] != tup:
            instance_triple_g.append(tup)
            instance_scope_g.append([pathi,pathi])
        instance_scope_g[len(instance_triple_g) - 1][1] = pathi
        if (pathi+1) % 100 == 0:
            pr_out = 'initializing path evidences ... %0.3f' % ((1.0 + pathi)/total_path)
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
            sen_word_cross[pathi][i] = word2id['BLANK']
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
                sen_word_cross[pathi][i] = word2id['UNK']
            else:
                sen_word_cross[pathi][i] = word2id[word]

            sen_pos1_cross[pathi][i] = pos_embed(i - en1pos_cross)
            sen_pos2_cross[pathi][i] = pos_embed(i - en2pos_cross)
    sys.stdout.write('\n')
    
    assert len(d_tup_lsent) == len(d_tup_lpath)
    assert len(instance_scope) == len(instance_scope_g)
        
    return (np.array(instance_triple), np.array(instance_scope), sen_len, sen_label, sen_word, sen_pos1, sen_pos2, sen_mask,
            sen_head, sen_tail, sen_head_type, sen_tail_type, sen_head_tying, sen_tail_tying,
            sen_word_cross, sen_pos1_cross, sen_pos2_cross, np.array(instance_scope_g))

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
    print kg_tup.shape
    print kg_tup_tying.shape
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
    print kg_tup_neg.shape
    print kg_tup_neg_tying.shape
    
    return np.array(kg_word), kg_tup, kg_tup_neg, kg_tup_tying, kg_tup_neg_tying

if __name__ == '__main__':
    start_time = time()
    print 'Initializing words and relations ...'
    
    relation2id = init_relation(export_path + 'relation2id.pkl')
    f_word2id = open(data_path + 'word2id.pkl', 'rb')
    word2id = pickle.load(f_word2id)
    f_word2id.close()
    
    sfout_vec = export_path + 'vec.npy'
    fout_vec = open(sfout_vec, 'wb')
    vec = np.load(data_path + 'vec.npy')
    np.save(sfout_vec, vec)
    sort_files('train0', 1)
    sort_files('test0', 1)

    
    data_type = 'training'
    print 'Initializing training data ...'
    if data_type == 'training':
        for i in [0]:
            (instance_triple, instance_scope, train_len, train_label, train_word, train_pos1, train_pos2, train_mask,
             train_head, train_tail, train_head_type, train_tail_type, train_head_tying, train_tail_tying,
             train_word_cross, train_pos1_cross, train_pos2_cross, instance_scope_path)= init_train_files("train%s_sort" % i)
            np.save(export_path+'train_instance_triple%s' % i, instance_triple)
            np.save(export_path+'train_instance_scope%s' % i, instance_scope)
            np.save(export_path+'train_len%s' % i, train_len)
            np.save(export_path+'train_label%s' % i, train_label)
            np.save(export_path+'train_word%s' % i, train_word)
            np.save(export_path+'train_pos1%s' % i, train_pos1)
            np.save(export_path+'train_pos2%s' % i, train_pos2)
            np.save(export_path+'train_mask%s' % i, train_mask)
            np.save(export_path+'train_head%s' % i, train_head)
            np.save(export_path+'train_tail%s' % i, train_tail)
            np.save(export_path+'train_head_type%s' % i, train_head_type)
            np.save(export_path+'train_tail_type%s' % i, train_tail_type)
            np.save(export_path+'train_head_tying%s' % i, train_head_tying)
            np.save(export_path+'train_tail_tying%s' % i, train_tail_tying)
            np.save(export_path+'train_word_cross%s' % i, train_word_cross)
            np.save(export_path+'train_pos1_cross%s' % i, train_pos1_cross)
            np.save(export_path+'train_pos2_cross%s' % i, train_pos2_cross)
            np.save(export_path+'train_instance_scope_path%s' % i, instance_scope_path)

    data_type = 'testing'
    print 'Initializing teting data ...'
    if data_type == 'testing':
        for i in [0]:
            (instance_triple, instance_scope, test_len, test_label, test_word, test_pos1, test_pos2, test_mask,
             test_head, test_tail, test_head_type, test_tail_type, test_head_tying, test_tail_tying,
             test_word_cross, test_pos1_cross, test_pos2_cross, instance_scope_path)= init_train_files("test%s_sort" % str(i))
            np.save(export_path+'test_instance_triple%s' % i, instance_triple)
            np.save(export_path+'test_instance_scope%s' % i, instance_scope)
            np.save(export_path+'test_len%s' % i, test_len)
            np.save(export_path+'test_label%s' % i, test_label)
            np.save(export_path+'test_word%s' % i, test_word)
            np.save(export_path+'test_pos1%s' % i, test_pos1)
            np.save(export_path+'test_pos2%s' % i, test_pos2)
            np.save(export_path+'test_mask%s' % i, test_mask)
            np.save(export_path+'test_head%s' % i, test_head)
            np.save(export_path+'test_tail%s' % i, test_tail)
            np.save(export_path+'test_head%s_type' % i, test_head_type)
            np.save(export_path+'test_tail%s_type' % i, test_tail_type)
            np.save(export_path+'test_head%s_tying' % i, test_head_tying)
            np.save(export_path+'test_tail%s_tying' % i, test_tail_tying)
            np.save(export_path+'test_word_cross%s' % i, test_word_cross)
            np.save(export_path+'test_pos1_cross%s' % i, test_pos1_cross)
            np.save(export_path+'test_pos2_cross%s' % i, test_pos2_cross)
            np.save(export_path+'test_instance_scope_path%s' % i, instance_scope_path)

    data_type = 'KG'
    print 'Initializing KG data ...'
    if data_type == 'KG':
        kg_word, kg_tup, kg_tup_neg, kg_tup_tying, kg_tup_neg_tying = init_kg()
        np.save(export_path+'kg_word', kg_word)
        np.save(export_path+'kg_tuple', kg_tup)
        np.save(export_path+'kg_tuple_neg', kg_tup_neg)
        np.save(export_path+'kg_tuple_tying', kg_tup_tying)
        np.save(export_path+'kg_tuple_neg_tying', kg_tup_neg_tying)
    
    end_time = time()
    time_taken = end_time - start_time
    hours, rest = divmod(time_taken, 3600)
    minutes, seconds = divmod(time_taken, 60)

    print 'Duration: about %s hours %s' % (hours, rest)
    print 'Duration: about %s minutes %s' % (minutes, seconds)
    
    
