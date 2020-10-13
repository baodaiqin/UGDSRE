import numpy as np
import pickle
import random
import sys

data_path = './origin_data_nyt10/'
export_path = './nyt10_part1/'

maxlen = 120
fixlen = 120
kg_fixlen = 20

def pos_embed(x):
    return max(0, min(x + maxlen, maxlen + maxlen + 1))

def init_word(sfout_word2ind, sfout_vec):
    all_w = []
    word2id, vec = load_w2v() 
    
    sf_train = data_path + 'text/train.txt'
    sf_test = data_path + 'text/test.txt'

    sf_kg = data_path + 'kg/train.txt'

    nb_line = 0
    with open(sf_train, 'rb') as fle:
        for line in fle:
            nb_line += 1
            if nb_line % 10000 == 0:
                pr_out = 'processing %s lines' % nb_line
                sys.stdout.write(pr_out+'\r')
                sys.stdout.flush()
            line = line.strip('\n')
            content = line.split('\t')
            try:
                e1 = content[0]
                e2 = content[1]
                sent = content[5]
                cross = content[6]
                cross3 = content[7]
                cross4 = content[8]
                all_w.append(e1)
                all_w.append(e2)
            except IndexError:
                continue

    with open(sf_test, 'rb') as fle:
        for line in fle:
            nb_line += 1
            if nb_line % 10000 == 0:
                pr_out = 'processing %s lines' % nb_line
		sys.stdout.write(pr_out+'\r')
                sys.stdout.flush()
            line = line.strip('\n')
            content = line.split('\t')
            try:
                e1 = content[0]
                e2 = content[1]
                all_w.append(e1)
                all_w.append(e2)
		sent = content[5]
                cross = content[6]
                cross3 = content[7]
                cross4 = content[8]
            except IndexError:
                continue
                
    with open(sf_kg, 'rb') as fle:
        for line in fle:
            line = line.strip('\n')
            try:
                e1, e2, srl = line.split('\t')
            except ValueError:
                continue
            
            all_w.append(e1)
            all_w.append(e2)

    all_w = list(set(all_w))
    all_w = sorted(all_w)
    all_w.extend(['PADDING', '<s1>', '</s1>', '<s2>', '</s2>', '<s3>', '</s3>', '<e1>', '<e2>'])
    all_w.append('UNK')
    all_w.append('BLANK')
    for w in all_w:
        if w not in word2id:
            wi = len(word2id)
            word2id[w] = wi

    fout_word2ind = open(sfout_word2ind, 'wb')
    pickle.dump(word2id, fout_word2ind)
    fout_word2ind.close()

    nb_vec, dim_vec = vec.shape
    nvec = np.ones((len(word2id)-nb_vec, dim_vec), dtype = np.float32)
    new_vec = np.concatenate((vec, nvec), axis=0)
    fout_vec = open(sfout_vec, 'wb')
    np.save(fout_vec, new_vec)
    fout_vec.close()

def load_w2v():
    word2ind = {}
    f = open(data_path + 'text/vec.txt', "r")
    total, size = f.readline().strip().split()[:2]
    total = (int)(total)
    word_size = (int)(size)
    vec = np.ones((total, word_size), dtype = np.float32)
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

    nb_nw = 0
    with open(data_path + 'word2id.txt', 'rb') as fle:
        for line in fle:
            line = line.strip('\n')
            try:
                w, i = line.split('\t')
            except ValueError:
                continue
            if w not in word2ind:
                wi = len(word2ind)
                word2ind[w] = wi
                nb_nw += 1

    nvec = np.ones((nb_nw, word_size), dtype = np.float32)
    vec = np.concatenate((vec, nvec), axis=0)
            
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

    with open(data_path + 'kg/train.txt', 'rb') as fle:
        for line in fle:
            line = line.strip('\n')
            try:
                e1, e2, rl = line.split('\t')
            except ValueError:
                continue
            if rl not in relation2id:
                rli = len(relation2id)
                relation2id[rl] = rli
    fout = open(sfout, 'wb')
    pickle.dump(relation2id, fout)
    fout.close()
    return relation2id
    
def init_train_files(name):
    f = open(data_path + 'text/' +  name + '.txt','r')
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
            sen_head, sen_tail, sen_word_cross, sen_pos1_cross, sen_pos2_cross, np.array(instance_scope_g))

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
        
        try:
            relation2id[rl]
            all_tup.append(tup)
        except KeyError:
            continue
    f.close()

    total = len(all_tup)
    kg_tup = np.zeros((total, 3), dtype = np.int32)
    kg_tup = []
    kg_tup_neg = []
    
    d_tup = {}
    all_e1 = []
    all_e2 = []
    
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
        except KeyError:
            pass

    kg_tup = np.array(kg_tup, dtype=np.int32)
    all_e1 = list(set(all_e1))
    all_e2 = list(set(all_e2))

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

    kg_tup_neg = np.array(kg_tup_neg, dtype=np.int32)
    return kg_tup, kg_tup_neg

if __name__ == '__main__':
    print 'Initializing words and relations ...'
    sfout_vec = export_path + 'vec.npy'
    sfout_word2id = export_path + 'word2id.pkl'
    init_word(sfout_word2id, sfout_vec)
    relation2id = init_relation(export_path + 'relation2id.pkl')
    
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
         train_head, train_tail, train_word_cross, train_pos1_cross, train_pos2_cross, instance_scope_path)= init_train_files("train")
        np.save(export_path+'train_instance_triple', instance_triple)
        np.save(export_path+'train_instance_scope', instance_scope)
        np.save(export_path+'train_len', train_len)
        np.save(export_path+'train_label', train_label)
        np.save(export_path+'train_word', train_word)
        np.save(export_path+'train_pos1', train_pos1)
        np.save(export_path+'train_pos2', train_pos2)
        np.save(export_path+'train_mask', train_mask)
        np.save(export_path+'train_head', train_head)
        np.save(export_path+'train_tail', train_tail)
        np.save(export_path+'train_word_cross', train_word_cross)
        np.save(export_path+'train_pos1_cross', train_pos1_cross)
        np.save(export_path+'train_pos2_cross', train_pos2_cross)
        np.save(export_path+'train_instance_scope_path', instance_scope_path)

    data_type = 'testing'
    print 'Initializing teting data ...'
    if data_type == 'testing':
        (instance_triple, instance_scope, test_len, test_label, test_word, test_pos1, test_pos2, test_mask,
         test_head, test_tail, test_word_cross, test_pos1_cross, test_pos2_cross, instance_scope_path)= init_train_files("test")
        np.save(export_path+'test_instance_triple', instance_triple)
        np.save(export_path+'test_instance_scope', instance_scope)
        np.save(export_path+'test_len', test_len)
        np.save(export_path+'test_label', test_label)
        np.save(export_path+'test_word', test_word)
        np.save(export_path+'test_pos1', test_pos1)
        np.save(export_path+'test_pos2', test_pos2)
        np.save(export_path+'test_mask', test_mask)
        np.save(export_path+'test_head', test_head)
        np.save(export_path+'test_tail', test_tail)
        np.save(export_path+'test_word_cross', test_word_cross)
        np.save(export_path+'test_pos1_cross', test_pos1_cross)
        np.save(export_path+'test_pos2_cross', test_pos2_cross)
        np.save(export_path+'test_instance_scope_path', instance_scope_path)

    data_type = 'KG'
    print 'Initializing KG data ...'
    if data_type == 'KG':
        kg_tup, kg_tup_neg = init_kg()
        np.save(export_path+'kg_tuple', kg_tup)
        np.save(export_path+'kg_tuple_neg', kg_tup_neg)
        
    
    
    
