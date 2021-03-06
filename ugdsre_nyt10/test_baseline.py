import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network_ug as network
import json
import sys
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from metrics import metrics
#import add_at_to_sent_type as add_at_to_sent
from kg_dataset_transe import KnowledgeGraph

export_path = "../nyt10_part1/"
export_path_g = "../nyt10_part2/"

word_vec = np.load(export_path + 'vec.npy')

KG = KnowledgeGraph(export_path)

FLAGS = tf.app.flags.FLAGS
path_evidence = sys.argv[1]
if path_evidence == 'sent':
        katt_flag = 10
elif path_evidence == 'sent_kg':
        katt_flag = 11

tf.app.flags.DEFINE_integer('nbatch_kg', 200,'entity numbers used each training time')
tf.app.flags.DEFINE_float('margin',1.0,'entity numbers used each training time')
tf.app.flags.DEFINE_float('learning_rate_kg',0.001,'learning rate for kg')
tf.app.flags.DEFINE_integer('rel_total', 1376,'total of relations')
tf.app.flags.DEFINE_integer('katt_flag', katt_flag, 'type of attention')

tf.app.flags.DEFINE_string('model', 'cnn', 'neural models to encode sentences')
tf.app.flags.DEFINE_integer('max_length', 120,'maximum of number of words in one sentence')
tf.app.flags.DEFINE_integer('pos_num', 120 * 2 + 1,'number of position embedding vectors')
tf.app.flags.DEFINE_integer('num_classes', 58,'maximum of relations')

tf.app.flags.DEFINE_integer('hidden_size', 230,'hidden feature size')
tf.app.flags.DEFINE_integer('pos_size',5,'position embedding size')

tf.app.flags.DEFINE_integer('max_epoch',30,'maximum of training epochs')
tf.app.flags.DEFINE_integer('batch_size', 2,'entity numbers used each training time')
tf.app.flags.DEFINE_float('learning_rate',0.05,'entity numbers used each training time')
tf.app.flags.DEFINE_float('weight_decay',0.00001,'weight_decay')
tf.app.flags.DEFINE_float('keep_prob',1.0,'dropout rate')

tf.app.flags.DEFINE_integer('test_batch_size', 2,'entity numbers used each test time')
tf.app.flags.DEFINE_string('checkpoint_path','./model_type_l/','path to store model')

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
        test_instance_scope_path = np.load(export_path_g + 'test_instance_scope_kg.npy')
        test_instance_scope_path3 = np.load(export_path_g + 'test_instance_scope_tx.npy')
        test_instance_scope_path4 = np.load(export_path_g + 'test_instance_scope_ug.npy')
	test_len = np.load(export_path + 'test_len.npy')
	test_label = np.load(export_path + 'test_label.npy')
	test_word = np.load(export_path + 'test_word.npy')
	test_pos1 = np.load(export_path + 'test_pos1.npy')
	test_pos2 = np.load(export_path + 'test_pos2.npy')

        test_word_cross = np.load(export_path_g + 'test_word_kg.npy')
        test_pos1_cross = np.load(export_path_g + 'test_pos1_kg.npy')
        test_pos2_cross = np.load(export_path_g + 'test_pos2_kg.npy')

        test_word_cross3 = np.load(export_path_g + 'test_word_tx.npy')
        test_pos1_cross3 = np.load(export_path_g + 'test_pos1_tx.npy')
        test_pos2_cross3 = np.load(export_path_g + 'test_pos2_tx.npy')

        test_word_cross4 = np.load(export_path_g + 'test_word_ug.npy')
        test_pos1_cross4 = np.load(export_path_g + 'test_pos1_ug.npy')
        test_pos2_cross4 = np.load(export_path_g + 'test_pos2_ug.npy')
        
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
                
	saver = tf.train.Saver()

	def test_step(head, tail, word, pos1, pos2, mask, leng, label_index, label, scope,
                      word_cr, pos1_cr, pos2_cr, scope_path, head_path, tail_path,
                      word_cr3, pos1_cr3, pos2_cr3, scope_path3, head_path3, tail_path3,
                      word_cr4, pos1_cr4, pos2_cr4, scope_path4, head_path4, tail_path4):
		feed_dict = {
			model.head_index: head,
			model.tail_index: tail,
                        model.head_index_path: head_path,
                        model.tail_index_path: tail_path,
                        model.head_index_path3: head_path3,
                        model.tail_index_path3: tail_path3,
                        model.head_index_path4: head_path4,
                        model.tail_index_path4: tail_path4,
			model.word: word,
			model.pos1: pos1,
			model.pos2: pos2,
                        model.word_cross: word_cr,
                        model.pos1_cross: pos1_cr,
                        model.pos2_cross: pos2_cr,
                        model.word_cross3: word_cr3,
                        model.pos1_cross3: pos1_cr3,
                        model.pos2_cross3: pos2_cr3,
                        model.word_cross4: word_cr4,
                        model.pos1_cross4: pos1_cr4,
                        model.pos2_cross4: pos2_cr4,
			model.mask: mask,
			model.len : leng,
			model.label_index: label_index,
			model.label: label,
			model.scope: scope,
                        model.scope_path: scope_path,
                        model.scope_path3: scope_path3,
                        model.scope_path4: scope_path4,
			model.keep_prob: FLAGS.keep_prob
		}
		output, test_att, test_pred = sess.run([model.test_output, model.test_att, model.test_pred], feed_dict)
		return output, test_att, test_pred

	f = open('results.txt','w')
	f.write('iteration\taverage precision\n')

        saver.restore(sess, './model_saved/cnn%s-274800' % FLAGS.katt_flag)
                
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
                input_scope_path3 = test_instance_scope_path3[i * FLAGS.test_batch_size:(i+1)*FLAGS.test_batch_size]
                input_scope_path4 = test_instance_scope_path4[i * FLAGS.test_batch_size:(i+1)*FLAGS.test_batch_size]
		index = []
                index_path = []
                index_path3 = []
                index_path4 = []
		scope = [0]
                scope_path = [0]
                scope_path3 = [0]
                scope_path4 = [0]
                
		label = []
                
                test_head_path = []
                test_tail_path = []
                test_head_path3 = []
                test_tail_path3 = []
                test_head_path4 = []
                test_tail_path4 = []
                        
		for num, num_path, num_path3, num_path4 in zip(input_scope, input_scope_path, input_scope_path3, input_scope_path4):
			index = index + range(num[0], num[1] + 1)
			label.append(test_label[num[0]])
			scope.append(scope[len(scope)-1] + num[1] - num[0] + 1)
                        
                        index_path = index_path + range(num_path[0], num_path[1] + 1)
                        scope_path.append(scope_path[len(scope_path)-1] + num_path[1] - num_path[0] + 1)

                        index_path3 = index_path3 + range(num_path3[0], num_path3[1] + 1)
                        scope_path3.append(scope_path3[len(scope_path3)-1] + num_path3[1] - num_path3[0] + 1)

                        index_path4 = index_path4 + range(num_path4[0], num_path4[1] + 1)
                        scope_path4.append(scope_path4[len(scope_path4)-1] + num_path4[1] - num_path4[0] + 1)
                        
                        test_head_path += [test_head[num[0]]]*len(range(num_path[0], num_path[1] + 1))
                        test_tail_path += [test_tail[num[0]]]*len(range(num_path[0], num_path[1] + 1))

                        test_head_path3 += [test_head[num[0]]]*len(range(num_path3[0], num_path3[1] + 1))
                        test_tail_path3 += [test_tail[num[0]]]*len(range(num_path3[0], num_path3[1] + 1))

                        test_head_path4 += [test_head[num[0]]]*len(range(num_path4[0], num_path4[1] + 1))
			test_tail_path4 += [test_tail[num[0]]]*len(range(num_path4[0], num_path4[1] + 1))
                                
		label_ = np.zeros((FLAGS.test_batch_size, FLAGS.num_classes))
		label_[np.arange(FLAGS.test_batch_size), label] = 1
		output, test_att, test_pred = test_step(test_head[index], test_tail[index], test_word[index,:], test_pos1[index,:], test_pos2[index,:],
                                                        test_mask[index,:], test_len[index], test_label[index], label_, np.array(scope),
                                                        test_word_cross[index_path,:], test_pos1_cross[index_path,:], test_pos2_cross[index_path,:],
                                                        np.array(scope_path), test_head_path, test_tail_path,
                                                        test_word_cross3[index_path3,:], test_pos1_cross3[index_path3,:], test_pos2_cross3[index_path3,:],
                                                        np.array(scope_path3), test_head_path3, test_tail_path3,
                                                        test_word_cross4[index_path4,:], test_pos1_cross4[index_path4,:], test_pos2_cross4[index_path4,:],
                                                        np.array(scope_path4), test_head_path4, test_tail_path4)
                                                                
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
        
        score = metrics(stack_label[:,1:], stack_output[:,1:])
        score.precision_at_k([100, 200, 300, 500, 1000, 2000])
                
        y_pred = np.argmax(stack_output, axis=1)
        y_true = np.argmax(stack_label, axis=1)
        print 'Precision: %s' % precision_score(y_true, y_pred, labels=range(1, stack_label.shape[1]), average='micro')
        print 'Recall: %s' % recall_score(y_true, y_pred, labels=range(1, stack_label.shape[1]), average='micro')
        print 'F1: %s' % f1_score(y_true, y_pred, labels=range(1, stack_label.shape[1]), average='micro')
        
	average_precision = average_precision_score(exclude_na_flatten_label,exclude_na_flatten_output, average = "micro")
                
        if int(FLAGS.katt_flag) == 10:
                np.save('./result/'+FLAGS.model+'+sent_prob'+'.npy', exclude_na_flatten_output)
		np.save('./result/'+FLAGS.model+'+sent_label'+'.npy',exclude_na_flatten_label)
        if int(FLAGS.katt_flag) == 11:
                np.save('./result/'+FLAGS.model+'+sent_kg_prob'+'.npy', exclude_na_flatten_output)
                np.save('./result/'+FLAGS.model+'+sent_kg_label'+'.npy',exclude_na_flatten_label)
                
	print 'pr: '+str(average_precision)
	f.write(str(average_precision)+'\n')
	f.close()

if __name__ == "__main__":
	tf.app.run()
