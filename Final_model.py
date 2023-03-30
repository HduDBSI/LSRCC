''''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
version:
Parallelized sampling on CPU
C++ evaluation for top-k recommendation
'''

import os
import sys

import tensorflow as tf
from tensorflow.python.client import device_lib
from helper import *
from time import time
import numpy as np
import scipy.sparse as sp
from parser import parse_args
import tensorflow.contrib as contrib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

cpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'CPU']
args = parse_args()

class LightGCN(object):
    def __init__(self,data_config_1, pretrain_data_1,data_config_2, pretrain_data_2,data_config_3, pretrain_data_3):
        # argument settings
        self.model_type = 'Ourmodel_three'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type

        self.pretrain_data_1= pretrain_data_1

        self.n_users_1= data_config_1['n_users']
        self.n_items_1= data_config_1['n_items']

        self.n_fold = 100

        self.norm_adj_1= data_config_1['norm_adj']
        self.n_nonzero_elems_1 = self.norm_adj_1.count_nonzero()

        self.pretrain_data_2 = pretrain_data_2

        self.n_users_2 = data_config_2['n_users']
        self.n_items_2 = data_config_2['n_items']

        self.norm_adj_2 = data_config_2['norm_adj']
        self.n_nonzero_elems_2 = self.norm_adj_2.count_nonzero()

        self.pretrain_data_3 = pretrain_data_3

        self.n_users_3 = data_config_3['n_users']
        self.n_items_3 = data_config_3['n_items']

        self.norm_adj_3 = data_config_3['norm_adj']
        self.n_nonzero_elems_3 = self.norm_adj_3.count_nonzero()

        self.temperature=0.2
        self.c_l=0.00001 #最优：0.0001
        self.global_loss_cl=0.00001


        self.lr = args.lr
        self.emb_dim = args.lightgcn_embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        #self.log_dir = self.create_model_str()
        self.verbose = args.verbose
        self.Ks = eval(args.Ks)

        #-------------------------------
        self.n_step_1 = 20
        self.n_step_2 = 50
        self.n_input = 64
        self.n_output = 64
        self.n_hidden = 30
        self.margin = 0.1
        self.batch_size = 1024
        self.learning_rate = 0.001

        self.mashups_embedding_1 = tf.placeholder(tf.float32, shape=[None, self.n_step_1, self.n_input])

        self.mashups_tag1 = tf.placeholder(tf.float32, shape=[None, self.n_input])
        self.mashups_tag2 = tf.placeholder(tf.float32, shape=[None, self.n_input])
        self.mashups_tag3 = tf.placeholder(tf.float32, shape=[None, self.n_input])
        self.mashups_tag4 = tf.placeholder(tf.float32, shape=[None, self.n_input])
        self.mashups_tag5 = tf.placeholder(tf.float32, shape=[None, self.n_input])

        self.item_embedding_1 = tf.placeholder(tf.float32, shape=[None, self.n_step_2, self.n_input])

        self.item_tag1 = tf.placeholder(tf.float32, shape=[None, self.n_input])
        self.item_tag2 = tf.placeholder(tf.float32, shape=[None, self.n_input])
        self.item_tag3 = tf.placeholder(tf.float32, shape=[None, self.n_input])
        self.item_tag4 = tf.placeholder(tf.float32, shape=[None, self.n_input])
        self.item_tag5 = tf.placeholder(tf.float32, shape=[None, self.n_input])

        # Define weights
        weights_1 = {
            # Hidden layer weights => 2*n_hidden because of forward + backward cells
            'out': tf.Variable(tf.random_normal([2 * self.n_hidden, self.n_output]))
        }
        biases_1 = {
            'out': tf.Variable(tf.random_normal([self.n_output]))
        }

        weights_mlp = {
            'in': tf.Variable(tf.random_normal([self.n_input, self.n_hidden])),
            'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_output]))
        }
        biases_mlp = {
            'in': tf.Variable(tf.random_normal([self.n_hidden])),
            'out': tf.Variable(tf.random_normal([self.n_output]))
        }

        with tf.variable_scope('mashup_1'):
            self.final_mashup_output_1=self.bilstm1(self.mashups_embedding_1,weights_1,biases_1)

        with tf.variable_scope('mashup_tag'):
            self.final_mashup_tag1=self.MLP(self.mashups_tag1,weights_mlp,biases_mlp)
            self.final_mashup_tag2 = self.MLP(self.mashups_tag2, weights_mlp, biases_mlp)
            self.final_mashup_tag3= self.MLP(self.mashups_tag3, weights_mlp, biases_mlp)
            self.final_mashup_tag4 = self.MLP(self.mashups_tag4, weights_mlp, biases_mlp)
            self.final_mashup_tag5 = self.MLP(self.mashups_tag5, weights_mlp, biases_mlp)

        self.final_mashup_tag=(self.final_mashup_tag1+self.final_mashup_tag2+self.final_mashup_tag3+
                               self.final_mashup_tag4+self.final_mashup_tag5)/5


        self.final_mashup_output_1=self.final_mashup_output_1+self.final_mashup_tag

        with tf.variable_scope('service_1'):
            self.final_service_output_1 = self.bilstm2(self.item_embedding_1, weights_1, biases_1)

        with tf.variable_scope('service_tag'):
            self.final_api_tag1 = self.MLP(self.item_tag1, weights_mlp, biases_mlp)
            self.final_api_tag2 = self.MLP(self.item_tag2, weights_mlp, biases_mlp)
            self.final_api_tag3 = self.MLP(self.item_tag3, weights_mlp, biases_mlp)
            self.final_api_tag4 = self.MLP(self.item_tag4, weights_mlp, biases_mlp)
            self.final_api_tag5 = self.MLP(self.item_tag5, weights_mlp, biases_mlp)

        self.final_api_tag=(self.final_api_tag1+self.final_api_tag2+self.final_api_tag3+
                            self.final_api_tag4+self.final_api_tag5)/5

        self.final_service_output_1=self.final_service_output_1+self.final_api_tag
        #self.final_service_output_1+self.final_api_tag

        #-------------------------------------------------

        # placeholder definition
        self.users_1 = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items_1 = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items_1= tf.placeholder(tf.int32, shape=(None,))

        self.users_2 = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items_2 = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items_2 = tf.placeholder(tf.int32, shape=(None,))

        self.users_3 = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items_3 = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items_3 = tf.placeholder(tf.int32, shape=(None,))

        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

        """
               *********************************************************
               Create Model Parameters (i.e., Initialize Weights).
               """
        # initialization of model parameters
        self.weights_1= self._init_weights(self.n_users_1, self.n_items_1, self.pretrain_data_1)

        self.weights_2 = self._init_weights(self.n_users_2, self.n_items_2, self.pretrain_data_2)

        self.weights_3 = self._init_weights(self.n_users_3, self.n_items_3, self.pretrain_data_3)

        """
                *********************************************************
                Compute Graph-based Representations of all users & items via Message-Passing Mechanism of Graph Neural Networks.
                Different Convolutional Layers:
                    1. ngcf: defined in 'Neural Graph Collaborative Filtering', SIGIR2019;
                    2. gcn:  defined in 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018;
                    3. gcmc: defined in 'Graph Convolutional Matrix Completion', KDD2018;
                """
        # if self.alg_type in ['lightgcn_1']:
        self.ua_embeddings_1, self.ia_embeddings_1 = self._create_lightgcn_embed(self.norm_adj_1, self.weights_1,
                                                                                 self.n_users_1, self.n_items_1)

        self.ua_embeddings_2, self.ia_embeddings_2 = self._create_lightgcn_embed(self.norm_adj_2, self.weights_2,
                                                                                 self.n_users_2, self.n_items_2)

        self.ua_embeddings_3, self.ia_embeddings_3 = self._create_lightgcn_embed(self.norm_adj_3, self.weights_3,
                                                                                 self.n_users_3, self.n_items_3)

        self.u_g_embeddings_1 = tf.nn.embedding_lookup(self.ua_embeddings_1, self.users_1)
        self.pos_i_g_embeddings_1 = tf.nn.embedding_lookup(self.ia_embeddings_1, self.pos_items_1)
        self.neg_i_g_embeddings_1 = tf.nn.embedding_lookup(self.ia_embeddings_1, self.neg_items_1)
        self.u_g_embeddings_pre_1 = tf.nn.embedding_lookup(self.weights_1['user_embedding'], self.users_1)
        self.pos_i_g_embeddings_pre_1 = tf.nn.embedding_lookup(self.weights_1['item_embedding'], self.pos_items_1)
        self.neg_i_g_embeddings_pre_1 = tf.nn.embedding_lookup(self.weights_1['item_embedding'], self.neg_items_1)

        self.u_g_embeddings_2 = tf.nn.embedding_lookup(self.ua_embeddings_2, self.users_2)
        self.pos_i_g_embeddings_2 = tf.nn.embedding_lookup(self.ia_embeddings_2, self.pos_items_2)
        self.neg_i_g_embeddings_2 = tf.nn.embedding_lookup(self.ia_embeddings_2, self.neg_items_2)
        self.u_g_embeddings_pre_2 = tf.nn.embedding_lookup(self.weights_2['user_embedding'], self.users_2)
        self.pos_i_g_embeddings_pre_2 = tf.nn.embedding_lookup(self.weights_2['item_embedding'], self.pos_items_2)
        self.neg_i_g_embeddings_pre_2 = tf.nn.embedding_lookup(self.weights_2['item_embedding'], self.neg_items_2)

        self.u_g_embeddings_3 = tf.nn.embedding_lookup(self.ua_embeddings_3, self.users_3)
        self.pos_i_g_embeddings_3 = tf.nn.embedding_lookup(self.ia_embeddings_3, self.pos_items_3)
        self.neg_i_g_embeddings_3 = tf.nn.embedding_lookup(self.ia_embeddings_3, self.neg_items_3)
        self.u_g_embeddings_pre_3 = tf.nn.embedding_lookup(self.weights_3['user_embedding'], self.users_3)
        self.pos_i_g_embeddings_pre_3 = tf.nn.embedding_lookup(self.weights_3['item_embedding'], self.pos_items_3)
        self.neg_i_g_embeddings_pre_3 = tf.nn.embedding_lookup(self.weights_3['item_embedding'], self.neg_items_3)

        """
                *********************************************************
                Generate Predictions & Optimize via BPR loss.
                """
        self.mf_loss_1, self.emb_loss_1, self.reg_loss_1 = self.create_bpr_loss(self.u_g_embeddings_1,
                                                                                self.pos_i_g_embeddings_1,
                                                                                self.neg_i_g_embeddings_1,
                                                                                self.u_g_embeddings_pre_1,
                                                                                self.pos_i_g_embeddings_pre_1,
                                                                                self.neg_i_g_embeddings_pre_1)
        self.loss_1 = self.mf_loss_1 + self.emb_loss_1

        self.mf_loss_2, self.emb_loss_2, self.reg_loss_2 = self.create_bpr_loss(self.u_g_embeddings_2,
                                                                                self.pos_i_g_embeddings_2,
                                                                                self.neg_i_g_embeddings_2,
                                                                                self.u_g_embeddings_pre_2,
                                                                                self.pos_i_g_embeddings_pre_2,
                                                                                self.neg_i_g_embeddings_pre_2)

        self.loss_2 = self.mf_loss_2 + self.emb_loss_2

        self.mf_loss_3, self.emb_loss_3, self.reg_loss_3 = self.create_bpr_loss(self.u_g_embeddings_3,
                                                                                self.pos_i_g_embeddings_3,
                                                                                self.neg_i_g_embeddings_3,
                                                                                self.u_g_embeddings_pre_3,
                                                                                self.pos_i_g_embeddings_pre_3,
                                                                                self.neg_i_g_embeddings_pre_3)

        self.loss_3 = self.mf_loss_3 + self.emb_loss_3

        self.loss_4 = self.bpr(self.final_mashup_output_1, self.final_service_output_1,
                               self.users_1, self.pos_items_1, self.neg_items_1)

        #self.global_loss=self.loss_4+self.global_loss_cl*(self.contrastive_loss3()+self.contrastive_loss4())

        self.global_loss = self.loss_4 + self.global_loss_cl * (self.contrastive_loss3_1() +self.contrastive_loss3_2()+
                                                                self.contrastive_loss4_1() +self.contrastive_loss4_2())

        #self.loss = self.loss_1+self.loss_2+self.loss_3+self.c_l*(self.contrastive_loss1()+self.contrastive_loss2())+self.global_loss
        self.loss = self.loss_1 + self.loss_2 + self.loss_3 +  self.global_loss

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _init_weights(self, n_users, n_items, pretrain_data):
        all_weights = dict()
        initializer = tf.random_normal_initializer(stddev=0.01)  # tf.contrib.layers.xavier_initializer()
        if pretrain_data is None:
            all_weights['user_embedding'] = tf.Variable(initializer([n_users, self.emb_dim]),
                                                        name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([n_items, self.emb_dim]),
                                                        name='item_embedding')
            print('using random initialization')  # print('using xavier initialization')
        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=pretrain_data['user_embed'], trainable=True,
                                                        name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=pretrain_data['item_embed'], trainable=True,
                                                        name='item_embedding', dtype=tf.float32)
            print('using pretrained initialization')

        self.weight_size_list = [self.emb_dim] + self.weight_size

        for k in range(self.n_layers):
            all_weights['W_gc_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_gc_%d' % k)

            all_weights['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

            all_weights['W_mlp_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_mlp_%d' % k)
            all_weights['b_mlp_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_mlp_%d' % k)

        return all_weights


    def bpr(self,mashup_output,service_output,mashups,pos,neg):

        u_e = tf.nn.embedding_lookup(mashup_output, mashups)
        pos_i_e = tf.nn.embedding_lookup(service_output, pos)
        neg_i_e = tf.nn.embedding_lookup(service_output, neg)

        pos_scores = tf.reduce_sum(tf.multiply(u_e, pos_i_e), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(u_e, neg_i_e), axis=1)

        regularizer = tf.nn.l2_loss(u_e) + tf.nn.l2_loss(pos_i_e) + tf.nn.l2_loss(neg_i_e)
        regularizer = regularizer / self.batch_size

        base_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
        base_loss = base_loss
        kge_loss = tf.constant(0.0, tf.float32, [1])
        reg_loss = 1e-5 * regularizer
        loss = base_loss + kge_loss + reg_loss

        return loss



    def bilstm1(self, x, weights, biases):

        x = tf.unstack(x, self.n_step_1, 1)

        lstm_fw_cell = contrib.rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
        lstm_bw_cell = contrib.rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)

        # Get lstm cell output
        try:
            outputs, _, _ = contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype = tf.float32)
        except Exception:  # Old TensorFlow version only returns outputs not states
            outputs = contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype = tf.float32)

        final_output = tf.matmul(outputs[-1], weights['out']) + biases['out']

        return final_output

    def bilstm2(self, x, weights, biases):

        x = tf.unstack(x, self.n_step_2, 1)

        lstm_fw_cell = contrib.rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
        lstm_bw_cell = contrib.rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)

        # Get lstm cell output
        try:
            outputs, _, _ = contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype = tf.float32)
        except Exception:  # Old TensorFlow version only returns outputs not states
            outputs = contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype = tf.float32)

        final_output = tf.matmul(outputs[-1], weights['out']) + biases['out']

        return final_output

    def bilstm3(self, x, weights, biases):

        x = tf.unstack(x, self.n_step_tag, 1)

        lstm_fw_cell = contrib.rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
        lstm_bw_cell = contrib.rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)

        # Get lstm cell output
        try:
            outputs, _, _ = contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype = tf.float32)
        except Exception:  # Old TensorFlow version only returns outputs not states
            outputs = contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype = tf.float32)

        final_output = tf.matmul(outputs[-1], weights['out']) + biases['out']

        return final_output

    def MLP(self, x, weights, biases):

        layer_addition = tf.matmul(x, weights['in']) + biases['in']
        layer_activation = tf.nn.tanh(layer_addition)
        hidden_drop = tf.nn.dropout(layer_activation, keep_prob=1.0)
        output = tf.matmul(hidden_drop, weights['out']) + biases['out']

        return output

    def _split_A_hat(self, X, n_users, n_items):
        A_fold_hat = []

        fold_len = (n_users + n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = n_users + n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X, n_users, n_items):
        A_fold_hat = []

        fold_len = (n_users + n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = n_users + n_items
            else:
                end = (i_fold + 1) * fold_len

            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def _create_lightgcn_embed(self, norm_adj, weights, n_users, n_items):
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(norm_adj, n_users, n_items)
        else:
            A_fold_hat = self._split_A_hat(norm_adj, n_users, n_items)

        ego_embeddings = tf.concat([weights['user_embedding'], weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [n_users, n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def create_bpr_loss(self, users, pos_items, neg_items, u_g_embeddings_pre, pos_i_g_embeddings_pre,
                        neg_i_g_embeddings_pre):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = tf.nn.l2_loss(u_g_embeddings_pre) + tf.nn.l2_loss(
            pos_i_g_embeddings_pre) + tf.nn.l2_loss(neg_i_g_embeddings_pre)
        regularizer = regularizer / self.batch_size

        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))

        emb_loss = self.decay * regularizer

        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)

    def contrastive_loss1(self):

        view1=self.ua_embeddings_1
        view2=self.ua_embeddings_2
        view1=tf.nn.l2_normalize(view1,1)
        view2=tf.nn.l2_normalize(view2,1)
        pos_score = tf.reduce_sum(tf.multiply(view1, view2), axis=1)
        # print(pos_score)
        ttl_score = tf.matmul(view1, view2, transpose_a=False, transpose_b=True)
        # print(ttl_score)
        pos_score = tf.exp(pos_score / self.temperature)
        ttl_score = tf.reduce_sum(tf.exp(ttl_score / self.temperature), axis=1)
        cl_loss = -tf.reduce_sum(tf.log(pos_score / ttl_score))
        # print(cl_loss)
        return cl_loss

    def contrastive_loss2(self):

        view1=self.ia_embeddings_1
        view2=self.ua_embeddings_3
        view1=tf.nn.l2_normalize(view1,1)
        view2=tf.nn.l2_normalize(view2,1)
        pos_score = tf.reduce_sum(tf.multiply(view1, view2), axis=1)
        # print(pos_score)
        ttl_score = tf.matmul(view1, view2, transpose_a=False, transpose_b=True)
        # print(ttl_score)
        pos_score = tf.exp(pos_score / self.temperature)
        ttl_score = tf.reduce_sum(tf.exp(ttl_score / self.temperature), axis=1)
        cl_loss = -tf.reduce_sum(tf.log(pos_score / ttl_score))
        # print(cl_loss)
        return cl_loss

    def contrastive_loss3(self):
        view1=self.final_mashup_output_1
        view2=tf.concat([self.ua_embeddings_1,self.ua_embeddings_2],axis=1) #局部 mashup1
        view1 = tf.nn.l2_normalize(view1, 1)
        view2 = tf.nn.l2_normalize(view2, 1)
        pos_score = tf.reduce_sum(tf.multiply(view1, view2), axis=1)
        # print(pos_score)
        ttl_score = tf.matmul(view1, view2, transpose_a=False, transpose_b=True)
        # print(ttl_score)
        pos_score = tf.exp(pos_score / self.temperature)
        ttl_score = tf.reduce_sum(tf.exp(ttl_score / self.temperature), axis=1)
        cl_loss = -tf.reduce_sum(tf.log(pos_score / ttl_score))
        # print(cl_loss)
        return cl_loss

    def contrastive_loss3_1(self):
        view1 = self.final_mashup_output_1
        view2 = self.ua_embeddings_1
        view1 = tf.nn.l2_normalize(view1, 1)
        view2 = tf.nn.l2_normalize(view2, 1)
        pos_score = tf.reduce_sum(tf.multiply(view1, view2), axis=1)
        # print(pos_score)
        ttl_score = tf.matmul(view1, view2, transpose_a=False, transpose_b=True)
        # print(ttl_score)
        pos_score = tf.exp(pos_score / self.temperature)
        ttl_score = tf.reduce_sum(tf.exp(ttl_score / self.temperature), axis=1)
        cl_loss = -tf.reduce_sum(tf.log(pos_score / ttl_score))
        # print(cl_loss)
        return cl_loss

    def contrastive_loss3_2(self):
        view1 = self.final_mashup_output_1
        view2 = self.ua_embeddings_2
        view1 = tf.nn.l2_normalize(view1, 1)
        view2 = tf.nn.l2_normalize(view2, 1)
        pos_score = tf.reduce_sum(tf.multiply(view1, view2), axis=1)
        # print(pos_score)
        ttl_score = tf.matmul(view1, view2, transpose_a=False, transpose_b=True)
        # print(ttl_score)
        pos_score = tf.exp(pos_score / self.temperature)
        ttl_score = tf.reduce_sum(tf.exp(ttl_score / self.temperature), axis=1)
        cl_loss = -tf.reduce_sum(tf.log(pos_score / ttl_score))
        # print(cl_loss)
        return cl_loss

    def contrastive_loss4(self):
        view1=self.final_service_output_1
        view2 = tf.concat((self.ia_embeddings_1, self.ua_embeddings_3), axis=1)  # 局部的 service1
        view1 = tf.nn.l2_normalize(view1, 1)
        view2 = tf.nn.l2_normalize(view2, 1)
        pos_score = tf.reduce_sum(tf.multiply(view1, view2), axis=1)
        # print(pos_score)
        ttl_score = tf.matmul(view1, view2, transpose_a=False, transpose_b=True)
        # print(ttl_score)
        pos_score = tf.exp(pos_score / self.temperature)
        ttl_score = tf.reduce_sum(tf.exp(ttl_score / self.temperature), axis=1)
        cl_loss = -tf.reduce_sum(tf.log(pos_score / ttl_score))
        # print(cl_loss)
        return cl_loss

    def contrastive_loss4_1(self):
        view1 = self.final_service_output_1
        view2 = self.ia_embeddings_1 # 局部的 service1
        view1 = tf.nn.l2_normalize(view1, 1)
        view2 = tf.nn.l2_normalize(view2, 1)
        pos_score = tf.reduce_sum(tf.multiply(view1, view2), axis=1)
        # print(pos_score)
        ttl_score = tf.matmul(view1, view2, transpose_a=False, transpose_b=True)
        # print(ttl_score)
        pos_score = tf.exp(pos_score / self.temperature)
        ttl_score = tf.reduce_sum(tf.exp(ttl_score / self.temperature), axis=1)
        cl_loss = -tf.reduce_sum(tf.log(pos_score / ttl_score))
        # print(cl_loss)
        return cl_loss

    def contrastive_loss4_2(self):
        view1 = self.final_service_output_1
        view2 = self.ua_embeddings_3 # 局部的 service1
        view1 = tf.nn.l2_normalize(view1, 1)
        view2 = tf.nn.l2_normalize(view2, 1)
        pos_score = tf.reduce_sum(tf.multiply(view1, view2), axis=1)
        # print(pos_score)
        ttl_score = tf.matmul(view1, view2, transpose_a=False, transpose_b=True)
        # print(ttl_score)
        pos_score = tf.exp(pos_score / self.temperature)
        ttl_score = tf.reduce_sum(tf.exp(ttl_score / self.temperature), axis=1)
        cl_loss = -tf.reduce_sum(tf.log(pos_score / ttl_score))
        # print(cl_loss)
        return cl_loss



