import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

FLAGS = tf.app.flags.FLAGS

class NN(object):

	def calc(self, e, t, r):
		return e + tf.reduce_sum(e * t, 1, keep_dims = True) * r

	def __init__(self, is_training, word_embeddings, simple_position = False):
		self.max_length = FLAGS.max_length
		self.num_classes = FLAGS.num_classes
		self.word_total, self.word_size = word_embeddings.shape
		self.hidden_size = FLAGS.hidden_size
                
		self.output_size = FLAGS.hidden_size

                self.margin = FLAGS.margin
		# placeholders for text models
		self.word = tf.placeholder(dtype=tf.int32,shape=[None, self.max_length], name='input_word')
		self.pos1 = tf.placeholder(dtype=tf.int32,shape=[None, self.max_length], name='input_pos1')
		self.pos2 = tf.placeholder(dtype=tf.int32,shape=[None, self.max_length], name='input_pos2')

                self.word_cross = tf.placeholder(dtype=tf.int32,shape=[None, self.max_length], name='input_word_cross')
                self.pos1_cross = tf.placeholder(dtype=tf.int32,shape=[None, self.max_length], name='input_pos1_cross')
                self.pos2_cross = tf.placeholder(dtype=tf.int32,shape=[None, self.max_length], name='input_pos2_cross')

                self.word_cross3 = tf.placeholder(dtype=tf.int32,shape=[None, self.max_length], name='input_word_cross3')
                self.pos1_cross3 = tf.placeholder(dtype=tf.int32,shape=[None, self.max_length], name='input_pos1_cross3')
                self.pos2_cross3 = tf.placeholder(dtype=tf.int32,shape=[None, self.max_length], name='input_pos2_cross3')

                self.word_cross4 = tf.placeholder(dtype=tf.int32,shape=[None, self.max_length], name='input_word_cross4')
                self.pos1_cross4 = tf.placeholder(dtype=tf.int32,shape=[None, self.max_length], name='input_pos1_cross4')
                self.pos2_cross4 = tf.placeholder(dtype=tf.int32,shape=[None, self.max_length], name='input_pos2_cross4')
                
		self.mask = tf.placeholder(dtype=tf.int32,shape=[None, self.max_length],name='input_mask')
		self.len = tf.placeholder(dtype=tf.int32,shape=[None],name='input_len')
		self.label_index = tf.placeholder(dtype=tf.int32,shape=[None], name='label_index')
		self.head_index = tf.placeholder(dtype=tf.int32,shape=[None], name='head_index')
		self.tail_index = tf.placeholder(dtype=tf.int32,shape=[None], name='tail_index')
                self.head_index_path = tf.placeholder(dtype=tf.int32,shape=[None], name='head_index_path')
                self.tail_index_path = tf.placeholder(dtype=tf.int32,shape=[None], name='tail_index_path')
                self.head_index_path3 = tf.placeholder(dtype=tf.int32,shape=[None], name='head_index_path3')
                self.tail_index_path3 = tf.placeholder(dtype=tf.int32,shape=[None], name='tail_index_path3')
                self.head_index_path4 = tf.placeholder(dtype=tf.int32,shape=[None], name='head_index_path4')
                self.tail_index_path4 = tf.placeholder(dtype=tf.int32,shape=[None], name='tail_index_path4')
		self.label = tf.placeholder(dtype=tf.float32,shape=[FLAGS.batch_size, self.num_classes], name='input_label')
		self.scope = tf.placeholder(dtype=tf.int32,shape=[FLAGS.batch_size+1], name='scope')
                self.scope_path = tf.placeholder(dtype=tf.int32,shape=[FLAGS.batch_size+1], name='scope_path')
                self.scope_path3 = tf.placeholder(dtype=tf.int32,shape=[FLAGS.batch_size+1], name='scope_path3')
                self.scope_path4 = tf.placeholder(dtype=tf.int32,shape=[FLAGS.batch_size+1], name='scope_path4')
		self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
		self.weights = tf.placeholder(dtype=tf.float32,shape=[FLAGS.batch_size])
		# placeholders for kg models
		self.pos_h = tf.placeholder(tf.int32, [None])
		self.pos_t = tf.placeholder(tf.int32, [None])
		self.pos_r = tf.placeholder(tf.int32, [None])
		self.neg_h = tf.placeholder(tf.int32, [None])
		self.neg_t = tf.placeholder(tf.int32, [None])
		self.neg_r = tf.placeholder(tf.int32, [None])

                with tf.name_scope("embedding-layers"):
			# word embeddings
			self.word_embedding = tf.get_variable(initializer=word_embeddings,name = 'word_embedding',dtype=tf.float32)
                        #self.word_embedding_cross = tf.get_variable(initializer=word_embeddings,name = 'word_embedding_cross',dtype=tf.float32)
                        
                        if FLAGS.katt_flag in [10]:
			        self.relation_matrix = tf.get_variable('relation_matrix',[self.num_classes, self.output_size],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
                        elif FLAGS.katt_flag in [11, 12]:
                                self.relation_matrix = tf.get_variable('relation_matrix',[self.num_classes, self.output_size*2],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
                        if FLAGS.katt_flag in [13]:
                                self.relation_matrix = tf.get_variable('relation_matrix',[self.num_classes, self.output_size*4],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
                        self.bias = tf.get_variable('bias',[self.num_classes],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
			# position embeddings
			temp_pos1_embedding = tf.get_variable('temp_pos1_embedding',[FLAGS.pos_num,FLAGS.pos_size],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
			temp_pos2_embedding = tf.get_variable('temp_pos2_embedding',[FLAGS.pos_num,FLAGS.pos_size],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
			self.pos1_embedding = tf.concat([temp_pos1_embedding,tf.reshape(tf.constant(np.zeros(FLAGS.pos_size,dtype=np.float32)),[1, FLAGS.pos_size])],0)
			self.pos2_embedding = tf.concat([temp_pos2_embedding,tf.reshape(tf.constant(np.zeros(FLAGS.pos_size,dtype=np.float32)),[1, FLAGS.pos_size])],0)

                        # relation embeddings and the transfer matrix between relations and textual relations
			self.rel_embeddings = tf.get_variable(name = "rel_embedding", shape = [FLAGS.rel_total, self.word_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
			self.transfer_matrix = tf.get_variable("transfer_matrix", [self.output_size, self.word_size*1])
			self.transfer_bias = tf.get_variable('transfer_bias', [self.word_size*1], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
                        
                with tf.name_scope("embedding-lookup"):
			# textual embedding-lookup 
			input_word = tf.nn.embedding_lookup(self.word_embedding, self.word)
			input_pos1 = tf.nn.embedding_lookup(self.pos1_embedding, self.pos1)
			input_pos2 = tf.nn.embedding_lookup(self.pos2_embedding, self.pos2)
			self.input_embedding = tf.concat(values = [input_word, input_pos1, input_pos2], axis = 2)

                        input_word_cross = tf.nn.embedding_lookup(self.word_embedding, self.word_cross)
                        input_pos1_cross = tf.nn.embedding_lookup(self.pos1_embedding, self.pos1_cross)
                        input_pos2_cross = tf.nn.embedding_lookup(self.pos2_embedding, self.pos2_cross)
                        self.input_embedding_cross = tf.concat(values = [input_word_cross, input_pos1_cross, input_pos2_cross], axis = 2)

                        input_word_cross3 = tf.nn.embedding_lookup(self.word_embedding, self.word_cross3)
                        input_pos1_cross3 = tf.nn.embedding_lookup(self.pos1_embedding, self.pos1_cross3)
                        input_pos2_cross3 = tf.nn.embedding_lookup(self.pos2_embedding, self.pos2_cross3)
                        self.input_embedding_cross3 = tf.concat(values = [input_word_cross3, input_pos1_cross3, input_pos2_cross3], axis = 2)

                        input_word_cross4 = tf.nn.embedding_lookup(self.word_embedding, self.word_cross4)
                        input_pos1_cross4 = tf.nn.embedding_lookup(self.pos1_embedding, self.pos1_cross4)
                        input_pos2_cross4 = tf.nn.embedding_lookup(self.pos2_embedding, self.pos2_cross4)
                        self.input_embedding_cross4 = tf.concat(values = [input_word_cross4, input_pos1_cross4, input_pos2_cross4], axis = 2)
			# knowledge embedding-lookup 
			pos_h = tf.nn.embedding_lookup(self.word_embedding, self.pos_h)
			pos_t = tf.nn.embedding_lookup(self.word_embedding, self.pos_t)
			pos_r = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)

                        neg_h = tf.nn.embedding_lookup(self.word_embedding, self.neg_h)
			neg_t = tf.nn.embedding_lookup(self.word_embedding, self.neg_t)
			neg_r = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)

		with tf.name_scope("knowledge_graph"):
			pos = tf.reduce_sum(abs(pos_h + pos_r - pos_t), 1, keep_dims = True)
			neg = tf.reduce_sum(abs(neg_h + neg_r - neg_t), 1, keep_dims = True)
			#self.loss_kg = tf.reduce_sum(pos - neg + self.margin)
                        self.loss_kg = tf.reduce_sum(tf.norm(pos - neg + self.margin))
                        
        def transfer(self, x):
		res = tf.nn.bias_add(tf.matmul(x, self.transfer_matrix), self.transfer_bias)
		return res

        def katt(self, x, is_training = True, dropout = True):
		with tf.name_scope("knowledge-based-attention"):
			head_e = tf.nn.embedding_lookup(self.word_embedding, self.head_index)
			tail_e = tf.nn.embedding_lookup(self.word_embedding, self.tail_index)
                        head = head_e
                        tail = tail_e
                        #kg_att = tf.concat([head, tail], axis=1)
                        kg_att = head - tail
                        #x = tf.layers.dropout(x, rate = self.keep_prob, training = is_training)
			attention_logit = tf.reduce_sum(self.transfer(x) * kg_att, 1)
			tower_repre = []
			for i in range(FLAGS.batch_size):
				sen_matrix = x[self.scope[i]:self.scope[i+1]]
				attention_score = tf.nn.softmax(tf.reshape(attention_logit[self.scope[i]:self.scope[i+1]], [1, -1]))
				final_repre = tf.reshape(tf.matmul(attention_score, sen_matrix),[self.output_size])
				tower_repre.append(final_repre)
			if dropout:
				stack_repre = tf.layers.dropout(tf.stack(tower_repre), rate = self.keep_prob, training = is_training)
			else:
				stack_repre = tf.stack(tower_repre)
		return stack_repre

        def gatt(self, xug, head_index, tail_index, scope, is_training = True, dropout = True):
                with tf.name_scope("knowledge-based-attention"):
                        head_e = tf.nn.embedding_lookup(self.word_embedding, head_index)
                        tail_e = tf.nn.embedding_lookup(self.word_embedding, tail_index)
                        head = head_e
                        tail = tail_e
                        #kg_att = tf.concat([head, tail], axis=1)
                        kg_att = head - tail
                        attention_logit_ug = tf.reduce_sum(self.transfer(xug) * kg_att, 1)
                        tower_repre = []
                        for i in range(FLAGS.batch_size):
                                ug_matrix = xug[scope[i]:scope[i+1]]
                                attention_logit_ug1 = attention_logit_ug[scope[i]:scope[i+1]]
                                attention_score = tf.nn.softmax(tf.reshape(attention_logit_ug1, [1, -1]))
                                final_repre = tf.reshape(tf.matmul(attention_score, ug_matrix),[self.output_size])
                                tower_repre.append(final_repre)
                        if dropout:
                                stack_repre = tf.layers.dropout(tf.stack(tower_repre), rate = self.keep_prob, training = is_training)
                        else:
                                stack_repre = tf.stack(tower_repre)
                return stack_repre

        def katt_test(self, x, is_training = False):
		head_t = tf.nn.embedding_lookup(self.word_embedding, self.head_index)
		tail_t = tf.nn.embedding_lookup(self.word_embedding, self.tail_index)
		head = head_t
                tail = tail_t
                #ht = tf.concat([head, tail], axis=1)
                ht = head - tail
                each_att = tf.expand_dims(ht, -1)
                kg_att = tf.concat([each_att for i in range(self.num_classes)], 2)
		x = tf.reshape(self.transfer(x), [-1, 1, self.word_size*1])
		test_attention_logit = tf.matmul(x, kg_att)
		return tf.reshape(test_attention_logit, [-1, self.num_classes])

        def gatt_test(self, x, head_index, tail_index, is_training = False):
                head_t = tf.nn.embedding_lookup(self.word_embedding, head_index)
                tail_t = tf.nn.embedding_lookup(self.word_embedding, tail_index)
                head = head_t
                tail = tail_t
                #ht = tf.concat([head, tail], axis=1)
                ht = head - tail
                each_att = tf.expand_dims(ht, -1)
                kg_att = tf.concat([each_att for i in range(self.num_classes)], 2)
                x = tf.reshape(self.transfer(x), [-1, 1, self.word_size*1])
                test_attention_logit = tf.matmul(x, kg_att)
                return tf.reshape(test_attention_logit, [-1, self.num_classes])

class CNN(NN):

	def __init__(self, is_training, word_embeddings, simple_position = False):
		NN.__init__(self, is_training, word_embeddings, simple_position)

		with tf.name_scope("conv-maxpool"):
			input_sentence = tf.expand_dims(self.input_embedding, axis=1)
			x = tf.layers.conv2d(inputs = input_sentence, filters=FLAGS.hidden_size, kernel_size=[1,3], strides=[1, 1], padding='same',
                                             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()) 
			x = tf.reduce_max(x, axis=2)
			x = tf.nn.relu(tf.squeeze(x))
                        
                        #with tf.name_scope("conv-maxpool-cross"):
                        input_sentence_cross = tf.expand_dims(self.input_embedding_cross, axis=1)
                        x_cross = tf.layers.conv2d(inputs = input_sentence_cross, filters=FLAGS.hidden_size, kernel_size=[1,3], strides=[1, 1], padding='same',
                                                   kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
                        x_cross = tf.reduce_max(x_cross, axis=2)
                        x_cross = tf.nn.relu(tf.squeeze(x_cross))

                        #with tf.name_scope("conv-maxpool-cross3"):
                        input_sentence_cross3 = tf.expand_dims(self.input_embedding_cross3, axis=1)
                        x_cross3 = tf.layers.conv2d(inputs = input_sentence_cross3, filters=FLAGS.hidden_size, kernel_size=[1,3], strides=[1, 1], padding='same',
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
                        x_cross3 = tf.reduce_max(x_cross3, axis=2)
                        x_cross3 = tf.nn.relu(tf.squeeze(x_cross3))

                        #with tf.name_scope("conv-maxpool-cross4"):
                        input_sentence_cross4 = tf.expand_dims(self.input_embedding_cross4, axis=1)
                        x_cross4 = tf.layers.conv2d(inputs = input_sentence_cross4, filters=FLAGS.hidden_size, kernel_size=[1,3], strides=[1, 1], padding='same',
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
                        x_cross4 = tf.reduce_max(x_cross4, axis=2)
                        x_cross4 = tf.nn.relu(tf.squeeze(x_cross4))
                        
                        
                if FLAGS.katt_flag == 10:
                        stack_repre_one = self.katt(x, is_training)
                        stack_repre = stack_repre_one
                if FLAGS.katt_flag == 11:
                        stack_repre_one = self.katt(x, is_training)
                        stack_repre_cross = self.gatt(x_cross, self.head_index_path, self.tail_index_path, self.scope_path, is_training)
                        #stack_repre = tf.reduce_max([stack_repre_one, stack_repre_cross], axis=0)
                        stack_repre = tf.concat([stack_repre_one, stack_repre_cross], axis=1)

                if FLAGS.katt_flag == 12:
                        stack_repre_one = self.katt(x, is_training)
                        stack_repre_cross3 = self.gatt(x_cross3, self.head_index_path3, self.tail_index_path3, self.scope_path3, is_training)
                        stack_repre = tf.concat([stack_repre_one, stack_repre_cross3], axis=1)
                        #stack_repre = tf.reduce_max([stack_repre_one, stack_repre_cross3], axis=0)
                        
                if FLAGS.katt_flag == 13:
                        stack_repre_one = self.katt(x, is_training)
                        stack_repre_cross = self.gatt(x_cross, self.head_index_path, self.tail_index_path, self.scope_path, is_training)
                        stack_repre_cross3 = self.gatt(x_cross3, self.head_index_path3, self.tail_index_path3, self.scope_path3, is_training)
                        stack_repre_cross4 = self.gatt(x_cross4, self.head_index_path4, self.tail_index_path4, self.scope_path4, is_training)
                        #stack_repre = tf.reduce_max([stack_repre_one, stack_repre_cross, stack_repre_cross3, stack_repre_cross4], axis=0)
                        stack_repre = tf.concat([stack_repre_one, stack_repre_cross, stack_repre_cross3, stack_repre_cross4], axis=1)
                
		with tf.name_scope("loss"):
			logits = tf.matmul(stack_repre, tf.transpose(self.relation_matrix)) + self.bias
			#self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label,logits=logits))
			self.loss = tf.losses.softmax_cross_entropy(onehot_labels = self.label, logits = logits, weights = self.weights)
			self.output = tf.nn.softmax(logits)
			tf.summary.scalar('loss',self.loss)
			self.predictions = tf.argmax(logits, 1, name="predictions")
			self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.label, 1))
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

		if not is_training:
			with tf.name_scope("test"):
                                if FLAGS.katt_flag == 10:
                                        test_attention_logit = self.katt_test(x)
                                elif FLAGS.katt_flag == 11:
                                        test_attention_logit = self.katt_test(x)
                                        test_attention_logit_cross = self.gatt_test(x_cross, self.head_index_path, self.tail_index_path)
                                elif FLAGS.katt_flag == 12:
                                        test_attention_logit = self.katt_test(x)
                                        test_attention_logit_cross3 = self.gatt_test(x_cross3, self.head_index_path3, self.tail_index_path3)
                                        
                                elif FLAGS.katt_flag == 13:
                                        test_attention_logit = self.katt_test(x)
                                        test_attention_logit_cross = self.gatt_test(x_cross, self.head_index_path, self.tail_index_path)
                                        test_attention_logit_cross3 = self.gatt_test(x_cross3, self.head_index_path3, self.tail_index_path3)
                                        test_attention_logit_cross4 = self.gatt_test(x_cross4, self.head_index_path4, self.tail_index_path4)
                                        
				test_tower_output = []
                                test_att_output = []
                                test_pred_output = []
                                test_pred_score = []
				for i in range(FLAGS.test_batch_size):
                                        if FLAGS.katt_flag == 10:
                                                test_attention_score = tf.nn.softmax(tf.transpose(test_attention_logit[self.scope[i]:self.scope[i+1],:]))
                                                final_repre_one = tf.matmul(test_attention_score, x[self.scope[i]:self.scope[i+1]])
                                                final_repre = final_repre_one
                                                logits = tf.matmul(final_repre, tf.transpose(self.relation_matrix)) + self.bias
                                                output = tf.diag_part(tf.nn.softmax(logits))
                                                test_att = tf.gather(test_attention_score, tf.argmax(output))
                                                test_att_output.append(test_att)
                                                test_pred_output.append(tf.argmax(output))
                                                test_pred_score.append(output)
                                                
                                        if FLAGS.katt_flag == 11:
                                                test_attention_score = tf.nn.softmax(tf.transpose(test_attention_logit[self.scope[i]:self.scope[i+1],:]))
                                                final_repre_one = tf.matmul(test_attention_score, x[self.scope[i]:self.scope[i+1]])
                                                
                                                test_attention_logit_cross_ = test_attention_logit_cross[self.scope_path[i]:self.scope_path[i+1],:]
                                                test_attention_score_cross_ = tf.nn.softmax(tf.transpose(test_attention_logit_cross_))
                                                
                                                x_cross_ = x_cross[self.scope_path[i]:self.scope_path[i+1]]
                                                final_repre_cross = tf.matmul(test_attention_score_cross_, x_cross_)
                                                final_repre = tf.concat([final_repre_one, final_repre_cross], axis=1)
                                                #final_repre = tf.reduce_max([final_repre_one, final_repre_cross], axis=0)

                                                logits = tf.matmul(final_repre, tf.transpose(self.relation_matrix)) + self.bias
                                                output = tf.diag_part(tf.nn.softmax(logits))
                                                test_att = tf.gather(test_attention_score, tf.argmax(output))
                                                test_att_output.append(test_att)
                                                test_pred_output.append(tf.argmax(output))
                                                test_pred_score.append(output)

                                        if FLAGS.katt_flag == 12:
                                                test_attention_score = tf.nn.softmax(tf.transpose(test_attention_logit[self.scope[i]:self.scope[i+1],:]))
                                                final_repre_one = tf.matmul(test_attention_score, x[self.scope[i]:self.scope[i+1]])
                                                
                                                test_attention_logit_cross3_ = test_attention_logit_cross3[self.scope_path3[i]:self.scope_path3[i+1],:]
                                                test_attention_score_cross3_ = tf.nn.softmax(tf.transpose(test_attention_logit_cross3_))

                                                x_cross3_ = x_cross3[self.scope_path3[i]:self.scope_path3[i+1]]
                                                final_repre_cross3 = tf.matmul(test_attention_score_cross3_, x_cross3_)
						final_repre = tf.concat([final_repre_one, final_repre_cross3], axis=1)
                                                #final_repre = tf.reduce_max([final_repre_one, final_repre_cross3], axis=0)

						logits = tf.matmul(final_repre, tf.transpose(self.relation_matrix)) + self.bias
                                                output = tf.diag_part(tf.nn.softmax(logits))
                                                test_att = tf.gather(test_attention_score, tf.argmax(output))
                                                test_att_output.append(test_att)
                                                test_pred_output.append(tf.argmax(output))
                                                test_pred_score.append(output)
                                                
                                        if FLAGS.katt_flag == 13:
                                                test_attention_score = tf.nn.softmax(tf.transpose(test_attention_logit[self.scope[i]:self.scope[i+1],:]))
                                                final_repre_one = tf.matmul(test_attention_score, x[self.scope[i]:self.scope[i+1]])
                                                
                                                test_attention_logit_cross_ = test_attention_logit_cross[self.scope_path[i]:self.scope_path[i+1],:]
                                                test_attention_score_cross_ = tf.nn.softmax(tf.transpose(test_attention_logit_cross_))
                                                test_attention_logit_cross3_ = test_attention_logit_cross3[self.scope_path3[i]:self.scope_path3[i+1],:]
                                                test_attention_score_cross3_ = tf.nn.softmax(tf.transpose(test_attention_logit_cross3_))
                                                test_attention_logit_cross4_ = test_attention_logit_cross4[self.scope_path4[i]:self.scope_path4[i+1],:]
                                                test_attention_score_cross4_ = tf.nn.softmax(tf.transpose(test_attention_logit_cross4_))

                                                x_cross_ = x_cross[self.scope_path[i]:self.scope_path[i+1]]
                                                final_repre_cross = tf.matmul(test_attention_score_cross_, x_cross_)
                                                x_cross3_ = x_cross3[self.scope_path3[i]:self.scope_path3[i+1]]
                                                final_repre_cross3 = tf.matmul(test_attention_score_cross3_, x_cross3_)
                                                x_cross4_ = x_cross4[self.scope_path4[i]:self.scope_path4[i+1]]
                                                final_repre_cross4 = tf.matmul(test_attention_score_cross4_, x_cross4_)
                                                
                                                #final_repre = tf.reduce_max([final_repre_one, final_repre_cross, final_repre_cross3, final_repre_cross4], axis=0)
                                                final_repre = tf.concat([final_repre_one, final_repre_cross, final_repre_cross3, final_repre_cross4], axis=1)
                                                
                                                logits = tf.matmul(final_repre, tf.transpose(self.relation_matrix)) + self.bias
                                                output = tf.diag_part(tf.nn.softmax(logits))
                                                test_att = tf.gather(test_attention_score, tf.argmax(output))
                                                test_att_output.append(test_att)
                                                test_pred_output.append(tf.argmax(output))
                                                test_pred_score.append(output)
                                                
					test_tower_output.append(output)
				test_stack_output = tf.reshape(tf.stack(test_tower_output),[FLAGS.test_batch_size, self.num_classes])
				self.test_output = test_stack_output
                                self.test_att = tf.concat(test_att_output, 0)
                                self.test_pred = tf.stack(test_pred_output)
                                self.test_sc = tf.stack(test_pred_score)
                                
                                
