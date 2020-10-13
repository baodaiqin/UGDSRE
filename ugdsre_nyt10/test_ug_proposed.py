import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network_pretrain_rank as network
import json
import sys
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from metrics import metrics
from kg_dataset_transe import KnowledgeGraph

#export_path = "/home/dq/EACL_gen_strict/data_type_year/"
#export_path_g = "/home/dq/EACL_gen_strict/data_type_g_year/"
export_path = "/home/dq/nyt10_part1/"
export_path_g = "/home/dq/nyt10_part1/" 

word_vec = np.load(export_path + 'vec.npy')

KG = KnowledgeGraph(export_path)

FLAGS = tf.app.flags.FLAGS
pre_or_not = sys.argv[1]

tf.app.flags.DEFINE_integer('nbatch_kg', 200,'entity numbers used each training time')
tf.app.flags.DEFINE_float('margin',1.0,'entity numbers used each training time')
tf.app.flags.DEFINE_float('learning_rate_kg',0.001,'learning rate for kg')
tf.app.flags.DEFINE_integer('rel_total', 1376,'total of relations')
tf.app.flags.DEFINE_integer('katt_flag', 13, 'type of attention')

tf.app.flags.DEFINE_string('model', 'cnn', 'neural models to encode sentences')
tf.app.flags.DEFINE_integer('max_length', 120,'maximum of number of words in one sentence')
tf.app.flags.DEFINE_integer('pos_num', 120 * 2 + 1,'number of position embedding vectors')
tf.app.flags.DEFINE_integer('num_classes', 58,'maximum of relations')

tf.app.flags.DEFINE_integer('hidden_size', 230,'hidden feature size')
tf.app.flags.DEFINE_integer('pos_size',5,'position embedding size')

tf.app.flags.DEFINE_integer('max_epoch', 100,'maximum of training epochs')
tf.app.flags.DEFINE_integer('batch_size', 2,'entity numbers used each training time')
tf.app.flags.DEFINE_float('learning_rate',0.05,'entity numbers used each training time')
tf.app.flags.DEFINE_float('weight_decay',0.00001,'weight_decay')
tf.app.flags.DEFINE_float('keep_prob',1.0,'dropout rate')

tf.app.flags.DEFINE_integer('test_batch_size', 2,'entity numbers used each test time')
tf.app.flags.DEFINE_string('checkpoint_path','./model_type_comp/','path to store model')

def complexity_features(array):
        return np.array([[np.count_nonzero(ele), np.unique(ele).size] for ele in array]).astype(np.int32)

def make_shape(array,last_dim):
	output = []
	for i in array:
		for j in i:
			output.append(j)
	output = np.array(output)
	if np.shape(output)[-1]==last_dim:
		return output

	else:
		print 'Make Shape Error!'

def main(_):
	print 'reading word embedding'
	word_vec = np.load(export_path + 'vec.npy')
	print 'reading test data'
	test_instance_triple = np.load(export_path + 'test_instance_triple.npy')
	test_instance_scope = np.load(export_path + 'test_instance_scope.npy')
        test_instance_scope_path = np.load(export_path_g + 'test_instance_scope_path.npy')
	test_len = np.load(export_path + 'test_len.npy')
	test_label = np.load(export_path + 'test_label.npy')
	test_word = np.load(export_path + 'test_word.npy')
	test_pos1 = np.load(export_path + 'test_pos1.npy')
	test_pos2 = np.load(export_path + 'test_pos2.npy')

        test_word_cross = np.load(export_path_g + 'test_word_cross.npy')
        test_pos1_cross = np.load(export_path_g + 'test_pos1_cross.npy')
        test_pos2_cross = np.load(export_path_g + 'test_pos2_cross.npy')

        test_comp_fea = complexity_features(test_word_cross)
        
        test_mask = np.load(export_path + 'test_mask.npy')
	test_head = np.load(export_path + 'test_head.npy')
	test_tail = np.load(export_path + 'test_tail.npy')
	print 'reading finished'
	print 'mentions 		: %d' % (len(test_instance_triple))
	print 'sentences		: %d' % (len(test_len))
	print 'relations		: %d' % (FLAGS.num_classes)
	print 'word size		: %d' % (len(word_vec[0]))
	print 'position size 	: %d' % (FLAGS.pos_size)
	print 'hidden size		: %d' % (FLAGS.hidden_size)
	print 'reading finished'

	print 'building network...'
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        
	if FLAGS.model.lower() == "cnn":
		model = network.CNN(is_training = False, word_embeddings = word_vec)
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()

	def test_step(head, tail, word, pos1, pos2, mask, leng, label_index, label, scope,
                      word_cr, pos1_cr, pos2_cr, scope_path, head_path, tail_path, comp_fea):
		feed_dict = {
			model.head_index: head,
			model.tail_index: tail,
                        model.head_index_path: head_path,
                        model.tail_index_path: tail_path,
			model.word: word,
			model.pos1: pos1,
			model.pos2: pos2,
                        model.word_cross: word_cr,
                        model.pos1_cross: pos1_cr,
                        model.pos2_cross: pos2_cr,
			model.mask: mask,
			model.len : leng,
			model.label_index: label_index,
			model.label: label,
			model.scope: scope,
                        model.scope_path: scope_path,
			model.keep_prob: FLAGS.keep_prob,
                        model.comp_fea: comp_fea
		}
		output, test_att, test_pred = sess.run([model.test_output, model.test_att, model.test_pred], feed_dict)
		return output, test_att, test_pred

	f = open('results.txt','w')
	f.write('iteration\taverage precision\n')
	
	if pre_or_not == 'ranking':
                saver.restore(sess, './model_saved/cnn13_re-322432')
        elif pre_or_not == 'ranking_pretrain':
                saver.restore(sess, './model_saved/cnn13_ALL-458000')
                        
	stack_output = []
	stack_label = []
        stack_att = []
        stack_pred = []
        stack_true = []
        stack_scope = []
		
	iteration = len(test_instance_scope)/FLAGS.test_batch_size
	for i in range(iteration):
		temp_str= 'running '+str(i)+'/'+str(iteration)+'...'
		sys.stdout.write(temp_str+'\r')
		sys.stdout.flush()
		input_scope = test_instance_scope[i * FLAGS.test_batch_size:(i+1)*FLAGS.test_batch_size]
                input_scope_path = test_instance_scope_path[i * FLAGS.test_batch_size:(i+1)*FLAGS.test_batch_size]
		index = []
                index_path = []
		scope = [0]
                scope_path = [0]
		label = []

                test_head_path = []
                test_tail_path = []
                        
		for num, num_path in zip(input_scope, input_scope_path):
			index = index + range(num[0], num[1] + 1)
			label.append(test_label[num[0]])
			scope.append(scope[len(scope)-1] + num[1] - num[0] + 1)
                                
                        index_path = index_path + range(num_path[0], num_path[1] + 1)
                        scope_path.append(scope_path[len(scope_path)-1] + num_path[1] - num_path[0] + 1)

                        test_head_path += [test_head[num[0]]]*len(range(num_path[0], num_path[1] + 1))
                        test_tail_path += [test_tail[num[0]]]*len(range(num_path[0], num_path[1] + 1))
                                
		label_ = np.zeros((FLAGS.test_batch_size, FLAGS.num_classes))
		label_[np.arange(FLAGS.test_batch_size), label] = 1
		output, test_att, test_pred = test_step(test_head[index], test_tail[index], test_word[index,:], test_pos1[index,:], test_pos2[index,:],
                                                        test_mask[index,:], test_len[index], test_label[index], label_, np.array(scope),
                                                        test_word_cross[index_path,:], test_pos1_cross[index_path,:], test_pos2_cross[index_path,:],
                                                        np.array(scope_path), test_head_path, test_tail_path, test_comp_fea[index_path,:])
                                                                
		stack_output.append(output)
		stack_label.append(label_)
                stack_att.append(test_att)
                stack_pred.append(test_pred)
                stack_true.extend(label)
                stack_scope.extend(input_scope)
			
	print 'evaluating...'
        
	stack_output = np.concatenate(stack_output, axis=0)
	stack_label = np.concatenate(stack_label, axis = 0)
        stack_att = np.concatenate(stack_att, axis=0)
        stack_pred = np.concatenate(stack_pred, axis=0)
        stack_true = np.array(stack_true)
        stack_scope = np.array(stack_scope)

	exclude_na_flatten_output = stack_output[:,1:]
	exclude_na_flatten_label = stack_label[:,1:]

        score = metrics(stack_label, stack_output)
        score.precision_at_k([100, 200, 300, 500, 1000, 2000])

        y_pred = np.argmax(stack_output, axis=1)
        y_true = np.argmax(stack_label, axis=1)
        print 'Precision: %s' % precision_score(y_true, y_pred, labels=range(1, stack_label.shape[1]), average='micro')
        print 'Recall: %s' % recall_score(y_true, y_pred, labels=range(1, stack_label.shape[1]), average='micro')
        print 'F1: %s' % f1_score(y_true, y_pred, labels=range(1, stack_label.shape[1]), average='micro')
                
	average_precision = average_precision_score(exclude_na_flatten_label,exclude_na_flatten_output, average = "micro")
        
        if pre_or_not == 'ranking':
		np.save('./result/'+FLAGS.model+'+sent_ug_ranking_prob'+'.npy', exclude_na_flatten_output)
                np.save('./result/'+FLAGS.model+'+sent_ug_ranking_label'+'.npy',exclude_na_flatten_label)
        elif pre_or_not == 'ranking_pretrain':
                np.save('./result/'+FLAGS.model+'+sen_ug_ranking_pretrain_rank_prob'+'.npy', exclude_na_flatten_output)
                np.save('./result/'+FLAGS.model+'+sen_ug_ranking_pretrain_rank_label'+'.npy',exclude_na_flatten_label)
                
	print 'AUC: '+str(average_precision)
	f.write(str(average_precision)+'\n')
	f.close()

if __name__ == "__main__":
	tf.app.run()
