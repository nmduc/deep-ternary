import tensorflow as tf

class ModelConfig():
    patch_size = 32             # image patch size
    sensing_rate = 0.25         # sensing rate
    hidden_sizes = [patch_size*patch_size*2, patch_size*patch_size*2]   # hidden layers' sizes
    dropout_keep_prob = 0       # 0 for no dropout
    use_bn = True               # to use batch-norm or not
    activation_fn = 'relu'      # activation function (relu or sigmoid)
    weight_spars_rate = 0.05    # sparsity ratio

class TrainConfig(object):
    scale = 1.0 / 68.639677345  # scale and mean for data normalization
    mean_value = 116.180734     # scale and mean should be calculated from your training data
    batch_size = 50             # batch size
    optimizer = "adam"          # can be "SGD", "momentum", "adam"
    initial_lr = 1e-2           # initial learning rate
    lr_update = 'step'          # 'val' or 'step'
    lr_decay_factor = 0.6       # learning rate decay factor
    num_epochs_per_decay = 5    # frequency of learning update (in terms of epoch)
    weight_decay = 1e-3         # weight for l2 regularization

def arr_to_string(arr):
    for i in xrange(len(arr)):
        arr[i] = str(arr[i])
    return ','.join(arr)

# model configs
tf.flags.DEFINE_float('patch_size', ModelConfig.patch_size,'')
tf.flags.DEFINE_float('sensing_rate', ModelConfig.sensing_rate,'')
tf.flags.DEFINE_string('hidden_sizes', arr_to_string(ModelConfig.hidden_sizes),'')
tf.flags.DEFINE_float('dropout_keep_prob', ModelConfig.dropout_keep_prob,'')
tf.flags.DEFINE_boolean('use_bn', ModelConfig.use_bn,'')
tf.flags.DEFINE_string('activation_fn', ModelConfig.activation_fn,'')
tf.flags.DEFINE_float('weight_spars_rate', ModelConfig.weight_spars_rate,'')

# training configs
tf.flags.DEFINE_boolean('scale', TrainConfig.scale,'')
tf.flags.DEFINE_float('mean_value', TrainConfig.mean_value,'')
tf.flags.DEFINE_integer('batch_size', TrainConfig.batch_size,'')
tf.flags.DEFINE_string('optimizer', TrainConfig.optimizer,'')
tf.flags.DEFINE_float('initial_lr', TrainConfig.initial_lr,'')
tf.flags.DEFINE_string('lr_update', TrainConfig.lr_update,'')
tf.flags.DEFINE_float('lr_decay_factor', TrainConfig.lr_decay_factor,'')
tf.flags.DEFINE_integer('num_epochs_per_decay', TrainConfig.num_epochs_per_decay,'')
tf.flags.DEFINE_float('weight_decay', TrainConfig.weight_decay,'')

CONFIGS = tf.app.flags.FLAGS
