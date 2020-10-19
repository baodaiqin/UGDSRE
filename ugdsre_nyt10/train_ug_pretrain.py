import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network_pretrain_rank as network
import network_pretrain as network_pre
import json
from sklearn.metrics import average_precision_score
import sys
import ctypes
import threading
from kg_dataset_transe import KnowledgeGraph

export_path = "/home/dq/nyt10_part1/"
export_path_g = "/home/dq/nyt10_part1/"
export_path_l = "/home/dq/nyt10_part2/"

word_vec = np.load(export_path + 'vec.npy')

KG = KnowledgeGraph(export_path)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('nbatch_kg', 200,'entity numbers used each training time')
tf.app.flags.DEFINE_float('margin',1.0,'entity numbers used each training time')
tf.app.flags.DEFINE_float('learning_rate_kg',0.001,'learning rate for kg')
tf.app.flags.DEFINE_integer('rel_total', 1376,'total of relations')
tf.app.flags.DEFINE_integer('katt_flag', 13, 'type of attention')

tf.app.flags.DEFINE_string('model', 'cnn', 'neural models to encode sentences')
tf.app.flags.DEFINE_integer('max_length', 120,'maximum of number of words in one sentence')
tf.app.flags.DEFINE_integer('pos_num', 120 * 2 + 1,'number of position embedding vectors')
tf.app.flags.DEFINE_integer('num_classes', 58,'maximum of relations')

tf.app.flags.DEFINE_integer('hidden_size', 230,'hidden feature size')
tf.app.flags.DEFINE_integer('pos_size', 5,'position embedding size')

#tf.app.flags.DEFINE_integer('max_epoch', 150,'maximum of training epochs')
tf.app.flags.DEFINE_integer('max_epoch', 2,'maximum of training epochs')
tf.app.flags.DEFINE_integer('max_epoch_pre', 1,'maximum of training epochs for pretrain')
tf.app.flags.DEFINE_integer('batch_size', 160,'entity numbers used each training time')
tf.app.flags.DEFINE_float('learning_rate',0.05,'learning rate for nn')
tf.app.flags.DEFINE_float('weight_decay',0.00001,'weight_decay')
tf.app.flags.DEFINE_float('keep_prob',0.5,'dropout rate')

tf.app.flags.DEFINE_string('model_dir','./model/','path to store model')
tf.app.flags.DEFINE_string('summary_dir','./summary','path to store summary_dir')

def complexity_features(array):
        return np.array([[np.count_nonzero(ele), np.unique(ele).size] for ele in array]).astype(np.float32)

def MakeSummary(name, value):
	"""Creates a tf.Summary proto with the given name and value."""
	summary = tf.Summary()
	val = summary.value.add()
	val.tag = str(name)
	val.simple_value = float(value)
	return summary

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

def main_pretrain(_):        
	word_vec = np.load(export_path + 'vec.npy')
	instance_triple = np.load(export_path + 'train_instance_triple.npy')
	instance_scope = np.load(export_path + 'train_instance_scope.npy')
        
        instance_scope_path = np.load(export_path_l + 'train_instance_scope_kg.npy')
        instance_scope_path3 = np.load(export_path_l + 'train_instance_scope_tx.npy')
        instance_scope_path4 = np.load(export_path_l + 'train_instance_scope_ug.npy')
        
        train_len = np.load(export_path + 'train_len.npy')
	train_label = np.load(export_path + 'train_label.npy')
	train_word = np.load(export_path + 'train_word.npy')
	train_pos1 = np.load(export_path + 'train_pos1.npy')
	train_pos2 = np.load(export_path + 'train_pos2.npy')

        train_word_cross = np.load(export_path_l + 'train_word_cross_kg.npy')
        train_pos1_cross = np.load(export_path_l + 'train_pos1_cross_kg.npy')
        train_pos2_cross = np.load(export_path_l + 'train_pos2_cross_kg.npy')

        train_word_cross3 = np.load(export_path_l + 'train_word_cross_tx.npy')
        train_pos1_cross3 = np.load(export_path_l + 'train_pos1_cross_tx.npy')
        train_pos2_cross3 = np.load(export_path_l + 'train_pos2_cross_tx.npy')

        train_word_cross4 = np.load(export_path_l + 'train_word_cross_ug.npy')
        train_pos1_cross4 = np.load(export_path_l + 'train_pos1_cross_ug.npy')
        train_pos2_cross4 = np.load(export_path_l + 'train_pos2_cross_ug.npy')
        
	train_mask = np.load(export_path + 'train_mask.npy')
	train_head = np.load(export_path + 'train_head.npy')
	train_tail = np.load(export_path + 'train_tail.npy')
        
	reltot = {}
	for index, i in enumerate(train_label):
		if not i in reltot:
			reltot[i] = 1.0
		else:
			reltot[i] += 1.0
	for i in reltot:
		reltot[i] = 1/(reltot[i] ** (0.05)) 
	print 'building network...'
	
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)

	model = network_pre.CNN(is_training = True, word_embeddings = word_vec)
	        
	global_step = tf.Variable(0,name='global_step',trainable=False)
	global_step_kg = tf.Variable(0,name='global_step_kg',trainable=False)
	tf.summary.scalar('learning_rate', FLAGS.learning_rate)
	tf.summary.scalar('learning_rate_kg', FLAGS.learning_rate_kg)

	optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
	grads_and_vars = optimizer.compute_gradients(model.loss)
	train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)

	optimizer_kg = tf.train.GradientDescentOptimizer(FLAGS.learning_rate_kg)
	grads_and_vars_kg = optimizer_kg.compute_gradients(model.loss_kg)
	train_op_kg = optimizer_kg.apply_gradients(grads_and_vars_kg, global_step = global_step_kg)
        
	merged_summary = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver(max_to_keep=None)
        
        batch_size = int(KG.n_triplet / FLAGS.nbatch_kg)

        
        def train_kg(coord):
		def train_step_kg(pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch):
			feed_dict = {
				model.pos_h: pos_h_batch,
				model.pos_t: pos_t_batch,
				model.pos_r: pos_r_batch,
				model.neg_h: neg_h_batch,
				model.neg_t: neg_t_batch,
				model.neg_r: neg_r_batch
			}
			_, step, loss = sess.run(
				[train_op_kg, global_step_kg, model.loss_kg], feed_dict)
			return loss

		batch_size = int(KG.n_triplet / FLAGS.nbatch_kg)
		times_kg = 0
                
		while not coord.should_stop():
			times_kg += 1
			res = 0.0
                        pos_batch_gen = KG.next_pos_batch(batch_size)
                        neg_batch_gen = KG.next_neg_batch(batch_size)
			for batchi in range(int(FLAGS.nbatch_kg)):
                                pos_batch = next(pos_batch_gen)
                                neg_batch = next(neg_batch_gen)
                                ph = pos_batch[:, 0]
                                pt = pos_batch[:, 1]
                                pr = pos_batch[:, 2]

                                nh = neg_batch[:, 0]
                                nt = neg_batch[:, 1]
                                nr = neg_batch[:, 2]

                                res += train_step_kg(ph, pt, pr, nh, nt, nr)
			time_str = datetime.datetime.now().isoformat()
			print "batch %d time %s | loss : %f" % (times_kg, time_str, res)
        
	def train_nn(coord):
		def train_step(head, tail, word, pos1, pos2, mask, leng, label_index, label, scope, weights,
                               word_cr, pos1_cr, pos2_cr, scope_path, head_path, tail_path):
                        
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
				model.weights: weights
			}
			_, step, loss, summary, output, correct_predictions = sess.run([train_op, global_step, model.loss, merged_summary, model.output, model.correct_predictions], feed_dict)
			summary_writer.add_summary(summary, step)
			return output, loss, correct_predictions

		stack_output = []
		stack_label = []
		stack_ce_loss = []

		train_order = range(len(instance_triple))

		save_epoch = 2
		eval_step = 300

		for one_epoch in range(FLAGS.max_epoch_pre):
			print('pretrain epoch '+str(one_epoch+1)+' starts!')
			np.random.shuffle(train_order)
			s1 = 0.0
			s2 = 0.0
			tot1 = 0.0
			tot2 = 1.0
			losstot = 0.0
			for i in range(int(len(train_order)/float(FLAGS.batch_size))):
                                #for i in range(50):
                                input_scope = np.take(instance_scope, train_order[i * FLAGS.batch_size:(i+1)*FLAGS.batch_size], axis=0)
                                input_scope_path = np.take(instance_scope_path, train_order[i * FLAGS.batch_size:(i+1)*FLAGS.batch_size], axis=0)
                                input_scope_path3 = np.take(instance_scope_path3, train_order[i * FLAGS.batch_size:(i+1)*FLAGS.batch_size], axis=0)
                                input_scope_path4 = np.take(instance_scope_path4, train_order[i * FLAGS.batch_size:(i+1)*FLAGS.batch_size], axis=0)
				index = []
				scope = [0]
                                index_path = []
                                index_path3 = []
                                index_path4 = []
                                scope_path = [0]
                                scope_path3 = [0]
                                scope_path4 = [0]
                                
				label = []
				weights = []

                                train_head_path = []
                                train_tail_path = []

                                train_head_path3 = []
                                train_tail_path3 = []

                                train_head_path4 = []
                                train_tail_path4 = []
                                
				for num, num_path, num_path3, num_path4 in zip(input_scope, input_scope_path, input_scope_path3, input_scope_path4):
					index = index + range(num[0], num[1] + 1)
					label.append(train_label[num[0]])
					scope.append(scope[len(scope)-1] + num[1] - num[0] + 1)
					weights.append(reltot[train_label[num[0]]])

                                        index_path = index_path + range(num_path[0], num_path[1] + 1)
                                        scope_path.append(scope_path[len(scope_path)-1] + num_path[1] - num_path[0] + 1)

                                        index_path3 = index_path3 + range(num_path3[0], num_path3[1] + 1)
                                        scope_path3.append(scope_path3[len(scope_path3)-1] + num_path3[1] - num_path3[0] + 1)

                                        index_path4 = index_path4 + range(num_path4[0], num_path4[1] + 1)
                                        scope_path4.append(scope_path4[len(scope_path4)-1] + num_path4[1] - num_path4[0] + 1)

                                        train_head_path += [train_head[num[0]]]*len(range(num_path[0], num_path[1] + 1))
                                        train_tail_path += [train_tail[num[0]]]*len(range(num_path[0], num_path[1] + 1))

                                        train_head_path3 += [train_head[num[0]]]*len(range(num_path3[0], num_path3[1] + 1))
                                        train_tail_path3 += [train_tail[num[0]]]*len(range(num_path3[0], num_path3[1] + 1))

                                        train_head_path4 += [train_head[num[0]]]*len(range(num_path4[0], num_path4[1] + 1))
                                        train_tail_path4 += [train_tail[num[0]]]*len(range(num_path4[0], num_path4[1] + 1))
                                        
				label_ = np.zeros((FLAGS.batch_size, FLAGS.num_classes))
				label_[np.arange(FLAGS.batch_size), label] = 1
                                
				output, loss, correct_predictions = train_step(train_head[index], train_tail[index], train_word[index,:], train_pos1[index,:], train_pos2[index,:],
                                                                               train_mask[index,:], train_len[index],train_label[index], label_, np.array(scope), weights,
                                                                               train_word_cross3[index_path3,:], train_pos1_cross3[index_path3,:], train_pos2_cross3[index_path3,:],
                                                                               np.array(scope_path3), train_head_path3, train_tail_path3)

                                output, loss, correct_predictions = train_step(train_head[index], train_tail[index], train_word[index,:], train_pos1[index,:], train_pos2[index,:],
                                                                               train_mask[index,:], train_len[index],train_label[index], label_, np.array(scope), weights,
                                                                               train_word_cross4[index_path4,:], train_pos1_cross4[index_path4,:], train_pos2_cross4[index_path4,:],
                                                                               np.array(scope_path4), train_head_path4, train_tail_path4)
                                
                                output, loss, correct_predictions = train_step(train_head[index], train_tail[index], train_word[index,:], train_pos1[index,:], train_pos2[index,:],
                                                                               train_mask[index,:], train_len[index],train_label[index], label_, np.array(scope), weights,
                                                                               train_word_cross[index_path,:], train_pos1_cross[index_path,:], train_pos2_cross[index_path,:],
                                                                               np.array(scope_path), train_head_path, train_tail_path)
				num = 0
				s = 0
				losstot += loss
				for num in correct_predictions:
					if label[s] == 0:
						tot1 += 1.0
						if num:
							s1+= 1.0
					else:
						tot2 += 1.0
						if num:
							s2 += 1.0
					s = s + 1

				time_str = datetime.datetime.now().isoformat()
				print "pretrain epoch %d step %d time %s | loss : %f, not NA accuracy: %f" % (one_epoch, i, time_str, loss, s2 / tot2)
				current_step = tf.train.global_step(sess, global_step)
                                
                        if (one_epoch + 1) % save_epoch == 0 and (one_epoch + 1) >= FLAGS.max_epoch_pre:
				print 'epoch '+str(one_epoch+1)+' has finished'
				print 'saving model...'
				path = saver.save(sess,FLAGS.model_dir+'pretrain_' + str(FLAGS.max_epoch_pre))
				print 'have savde model to '+path

		coord.request_stop()


	coord = tf.train.Coordinator()
	threads = []
        threads.append(threading.Thread(target=train_kg, args=(coord,)))
	threads.append(threading.Thread(target=train_nn, args=(coord,)))
	for t in threads: t.start()
	coord.join(threads)
        
if __name__ == "__main__":
	tf.app.run(main_pretrain) 
