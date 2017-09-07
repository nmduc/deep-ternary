import tensorflow as tf
from tensorflow.python import debug as tf_debug
import tensorflow.contrib.slim as slim
import tensorflow.contrib.losses as L
import numpy as np 

class Autoencoder():
    def __init__(self, x_dim, cfg, log_dir=None):
        self.parse_model_configs(cfg)
        self.x_dim = x_dim
        assert self.x_dim == self.patch_size * self.patch_size, 'Patch size and data dimension mis-match'
        self.y_dim = int(np.floor(self.x_dim * self.sensing_rate))

        # define placeholders
        self.x = tf.placeholder(tf.float32, [None, self.x_dim])
        self.lr = tf.placeholder(tf.float32, [])    # learning rate
        self.is_training = tf.placeholder(tf.bool, [], name='is_training')

        # initializer
        self.initializer = tf.contrib.layers.xavier_initializer 

        self.y, self.sen_w, self.sen_wsb, self.w_scale = self.build_sparse_binary_sensing_matrix(self.x, self.x_dim, self.y_dim, 'sensing')
        self.hidden = self.build_nonlinear_decoder(self.y, self.hidden_sizes, weight_decay=self.weight_decay, scope='decoder')
        self.x_hat = self.build_reconstructor(self.hidden, self.x_dim, weight_decay=self.weight_decay, scope='reconstructor')

        # loss functions
        self.recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.x_hat - self.x), axis=1))

        # build train_opts
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = self.build_optimizer(self.optimizer_name, self.lr)
        self.tvars = tf.trainable_variables()
        self.grads_vars = self.optimizer.compute_gradients(self.recon_loss, var_list=self.tvars)
        for grad_var in self.grads_vars:
            if grad_var[1].op.name == 'sensing/sensing_sparse_binary_weights':
                # replace gradients of the consinuous weights with those of the sparse binary weights
                self.grads_vars.remove(grad_var)
                self.grads_vars.append([grad_var[0], self.sen_w])
        self.train_opt = self.optimizer.apply_gradients(self.grads_vars, global_step=self.global_step)

        self.update_sen_wb_opt = self.build_weight_sparsify_binarize_opt(self.sen_wsb, self.sen_w)
        self.binary_weight_scale_update_opt = self.build_binary_weight_scale_update_opt(self.sen_w, self.w_scale)

        init = tf.global_variables_initializer()
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=sess_config)
        self.sess.run(init)
        self.prepare_logger(log_dir)

    def prepare_logger(self, log_dir):
        # summary writer
        self.saver = tf.train.Saver(max_to_keep=10)
        if log_dir:
            self.writer = tf.summary.FileWriter(log_dir, self.sess.graph)
            tf.summary.scalar("recon_loss", self.recon_loss)
            for var in self.tvars:
                tf.summary.histogram(var.op.name, var)
            for grad_var_pair in self.grads_vars:
                if grad_var_pair[0] is None:
                    continue
                tf.summary.histogram(grad_var_pair[0].op.name, grad_var_pair[0])
            self.merged_summaries = tf.summary.merge_all()

    def parse_model_configs(self, cfg):
        self.weight_decay = cfg.weight_decay
        self.optimizer_name = cfg.optimizer
        self.hidden_sizes = self.string_to_array(cfg.hidden_sizes, dtype='int')
        self.sensing_rate = cfg.sensing_rate
        self.patch_size = cfg.patch_size
        self.keep_prob = None
        if cfg.dropout_keep_prob > 0 and cfg.dropout_keep_prob <= 1:
            self.keep_prob = cfg.dropout_keep_prob
        self.use_bn = cfg.use_bn
        if cfg.activation_fn == 'relu':
            self.transfer = tf.nn.relu
        elif cfg.activation_fn == 'sigmoid':
            self.transfer = tf.nn.sigmoid
        self.weight_spars_rate = cfg.weight_spars_rate
        
    def build_sparse_binary_sensing_matrix(self, inp, inp_dim, out_dim, scope):
        with tf.variable_scope(scope) as scp:
            weights = tf.get_variable('%s_weights' %scope, shape=[inp_dim, out_dim], 
                initializer=self.initializer(), dtype=tf.float32, trainable=False)
            sparse_binary_weights = tf.get_variable('%s_sparse_binary_weights' %scope, shape=[inp_dim, out_dim], 
                initializer=None, dtype=tf.float32, trainable=True)
            scale = tf.get_variable('%s_binary_weight_scale' %scope, 
                initializer=np.ones((1, self.y_dim)).astype(np.float32), dtype=tf.float32, trainable=False)
            output = tf.matmul(inp, sparse_binary_weights)
            output = tf.multiply(output, scale)
            return output, weights, sparse_binary_weights, scale

    def build_nonlinear_decoder(self, inp, hidden_sizes, weight_decay, scope):
        hidden = inp
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.fully_connected], 
                weights_initializer=self.initializer(),
                biases_initializer=tf.constant_initializer(0),
                activation_fn=self.transfer,
                weights_regularizer=slim.l2_regularizer(weight_decay)):              
                for i in xrange(len(hidden_sizes)):
                    hidden = slim.fully_connected(hidden, hidden_sizes[i], scope='fc%d'%i)
                    if self.use_bn:
                        hidden = self.bn_layer(hidden, scope='bn%d'%i)
                    if self.keep_prob:
                        hidden = slim.dropout(hidden, self.keep_prob, 
                            is_training=self.is_training, scope='dropout%d'%i)
        return hidden

    def build_reconstructor(self, inp, output_dim, weight_decay, scope):
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.fully_connected], 
                weights_initializer=self.initializer(),
                biases_initializer=tf.constant_initializer(0),
                activation_fn=None,
                weights_regularizer=slim.l2_regularizer(weight_decay)): 
                x_hat = slim.fully_connected(inp, output_dim)
                if self.use_bn:
                    x_hat = self.bn_layer(x_hat, scope='bn')
        return x_hat

    def calc_spars_mask(self, tensor, spars_level):
        ''' spars_level: number of non-zero elements
        '''
        tensor = tf.transpose(tensor)
        abs_tensor = tf.abs(tensor)
        _, indices = tf.nn.top_k(abs_tensor, k=spars_level)
        
        range_ = tf.expand_dims(tf.range(0, indices.get_shape()[0]), 1)  # will be [[0], [1]]
        range_repeated = tf.tile(range_, [1, spars_level])  # will be [[0, 0, ...], [1, 1, ...]]

        full_indices = tf.concat([tf.expand_dims(range_repeated, 2), tf.expand_dims(indices, 2)], 2)  # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
        full_indices = tf.reshape(full_indices, [-1, 2])

        mask = tf.sparse_to_dense(full_indices, tensor.get_shape(), sparse_values=1., default_value=0., validate_indices=False)
        mask = tf.transpose(mask)
        return mask

    def build_weight_sparsify_binarize_opt(self, wsb, w):
        ''' build opt to sparsify and then binarize the weights
        '''
        spars_level = int(self.x_dim * self.weight_spars_rate)
        mask = self.calc_spars_mask(w, spars_level)
        # binarize the sparse weights
        sen_w_sparse_binary = tf.multiply(tf.sign(w), mask) 
        opt = tf.assign(wsb, sen_w_sparse_binary, validate_shape=True)
        return opt

    def build_binary_weight_scale_update_opt(self, w, scale):
        spars_level = int(self.x_dim * self.weight_spars_rate)
        mask = self.calc_spars_mask(w, spars_level)
        val = tf.reduce_sum(tf.multiply(tf.abs(w), mask), axis=0, keep_dims=True) / spars_level
        opt = tf.assign(scale, val)
        return opt

    def build_optimizer(self, optimizer_name, lr):
        # optimizer
        if optimizer_name == "SGD":
            optimizer = tf.train.GradientDescentOptimizer(lr)
        elif optimizer_name == 'momentum':
            optimizer = tf.train.MomentumOptimizer(lr, 0.9)
        elif optimizer_name == 'adam':
            optimizer = tf.train.AdamOptimizer(lr)
        assert optimizer, 'Invalid learning algorithm'
        return optimizer

    #-----------------------------------------------------------------------------------------
    # Operation functions
    #-----------------------------------------------------------------------------------------
    def quantize_weights(self):
        dummy_x = np.zeros((1, self.x_dim))
        self.sess.run([self.update_sen_wb_opt, self.binary_weight_scale_update_opt], 
                feed_dict={self.x:dummy_x, self.lr:0, self.is_training:True})

    def partial_fit(self, x, lr, get_summary=False):
        # train one step 
        summary = None
        step = self.sess.run(self.global_step)
        fd = {self.x:x, self.lr:lr, self.is_training:True}
        if get_summary:
            loss, train_opt, summary = self.sess.run([self.recon_loss, self.train_opt, self.merged_summaries], feed_dict=fd)
        else:
            loss, train_opt = self.sess.run([self.recon_loss, self.train_opt], feed_dict=fd)
        # quantize and/or sparsify sensing weights 
        self.sess.run([self.update_sen_wb_opt, self.binary_weight_scale_update_opt], feed_dict=fd)
        return loss, train_opt, summary, step

    def sense_x(self, x):
        y = self.sess.run((self.y), feed_dict={self.x:x, self.lr:0, self.is_training:False})
        return y

    def reconstruct_x(self, x):
        x_hat = self.sess.run((self.x_hat), feed_dict={self.x:x, self.lr:0, self.is_training:False})
        return x_hat

    def reconstruct(self, measurement):
        x_hat = self.sess.run((self.x_hat), feed_dict={self.y:measurement, self.lr:0, self.is_training:False})
        return x_hat

    def calc_loss(self, x):
        loss = self.sess.run(self.recon_loss, feed_dict={self.x:x, self.lr:0, self.is_training:False})
        return loss

    def get_sensing_matrix(self):
        w, sbw, sc = self.sess.run([self.sen_w,self.sen_wsb,self.w_scale])
        return w, sbw, sc

    #-----------------------------------------------------------------------------------------
    # END Operation functions
    #-----------------------------------------------------------------------------------------

    #-----------------------------------------------------------------------------------------
    # Util functions 
    #-----------------------------------------------------------------------------------------
    def bn_layer(self, inputs, scope):
        bn = tf.contrib.layers.batch_norm(inputs, is_training=self.is_training, 
            center=True, fused=False, scale=True, updates_collections=None, decay=0.9, scope=scope)
        return bn

    def save(self, save_path):
        self.saver.save(self.sess, save_path, global_step=self.sess.run(self.global_step))

    def restore(self, save_path):
        self.saver.restore(self.sess, save_path)

    def log(self, summary):
        self.writer.add_summary(summary, global_step=self.sess.run(self.global_step))

    def string_to_array(self, str, dtype='int'):
        arr = str.strip().split(',')
        for i in xrange(len(arr)):
            if dtype == 'int':
                arr[i] = int(arr[i])
            elif dtype == 'float':
                arr[i] = float(arr[i])
        return arr
    #-----------------------------------------------------------------------------------------
    # END Util functions
    #-----------------------------------------------------------------------------------------
