from utils.data_loader import DataLoader
import tensorflow as tf
import numpy as np 
import configs.configs as configs
from models.autoencoder import Autoencoder
import time 

FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string("db_fname", "./data/patches_32x32_2k.h5", "Path to database file used for training")
tf.flags.DEFINE_string("output_basedir", "./outputs/", "Directory for saving and loading model checkpoints")
tf.flags.DEFINE_string("pretrained_fname", "", "Name of the pretrained model checkpoints (to resume from)")
tf.flags.DEFINE_integer("n_epochs", 50, "Number of training epochs.")
tf.flags.DEFINE_integer("log_every_n_steps", 50,
                        "Frequency at which loss and global step are logged.")
tf.flags.DEFINE_integer("save_every_n_epochs", 10,
                        "Frequency at which session is saved.")
tf.flags.DEFINE_boolean("log_time", False, "Whether to print out running time or not")
tf.flags.DEFINE_integer("n_vals", 400, "Number of validation samples.")

FLAGS.output_dir = FLAGS.output_basedir + 'snapshots/snapshot'
FLAGS.log_dir = FLAGS.output_basedir + 'log/'

cfgs = configs.CONFIGS

def log_train(fout, msg):
    print(msg)
    fout.write('%s\n' %msg)

def val(model, data_loader):
    print('Perform validation ... ')
    total_loss = 0.0
    n_batches = 0
    while True:
        x, flag = data_loader.next_batch(cfgs.batch_size, 'val')
        n_batches += 1
        loss = model.calc_loss(x)
        total_loss += loss
        if flag:
            break
    print('Done')
    return total_loss / n_batches

def main(unused_argv):
    val_losses = []
    assert FLAGS.output_dir, "--output_dir is required"
    # Create training directory.
    output_dir = FLAGS.output_dir
    if not tf.gfile.IsDirectory(output_dir):
        tf.gfile.MakeDirs(output_dir)

    dl = DataLoader(FLAGS.db_fname, mean=cfgs.mean_value, scale=cfgs.scale, n_vals=FLAGS.n_vals)
    dl.prepare()

    x_dim = dl.get_data_dim()
    model = Autoencoder(x_dim, cfgs, log_dir=FLAGS.log_dir)
    model.quantize_weights()

    txt_log_fname = FLAGS.log_dir + 'text_log.txt'
    log_fout = open(txt_log_fname, 'w')

    if FLAGS.pretrained_fname:
        try:
            log_train(log_fout, 'Resume from %s' %(FLAGS.pretrained_fname))
            model.restore(FLAGS.pretrained_fname)
        except:
            log_train(log_fout, 'Cannot restore from %s' %(FLAGS.pretrained_fname))
            pass
    
    lr = cfgs.initial_lr
    epoch_counter = 0
    ite = 0
    while True:
        start = time.time()
        x, flag = dl.next_batch(cfgs.batch_size, 'train')
        load_data_time = time.time() - start
        if flag: 
            epoch_counter += 1
        
        do_log = (ite % FLAGS.log_every_n_steps == 0) or flag
        do_snapshot = flag and epoch_counter > 0 and epoch_counter % FLAGS.save_every_n_epochs == 0
        val_loss = -1

        # train one step
        start = time.time()
        loss, _, summary, ite = model.partial_fit(x, lr, do_log)
        one_iter_time = time.time() - start
        
        # writing outs
        if do_log:
            log_train(log_fout, 'Iteration %d, (lr=%f) training loss  : %f' %(ite, lr, loss))
            if FLAGS.log_time:
                log_train(log_fout, 'Iteration %d, data loading: %f(s) ; one iteration: %f(s)' 
                    %(ite, load_data_time, one_iter_time))
            model.log(summary)
        if flag:
            val_loss = val(model, dl)
            val_losses.append(val_loss)
            log_train(log_fout, '----------------------------------------------------')
            if ite == 0:
                log_train(log_fout, 'Initial validation loss: %f' %(val_loss))
            else:
                log_train(log_fout, 'Epoch %d, validation loss: %f' %(epoch_counter, val_loss))
            log_train(log_fout, '----------------------------------------------------')
            model.log(summary)
        if do_snapshot:
            log_train(log_fout, 'Snapshotting')
            model.save(FLAGS.output_dir)
        
        if flag: 
            if cfgs.lr_update == 'val' and len(val_losses) >= 5 and val_loss >= max(val_losses[-5:-1]):
                    lr = lr * cfgs.lr_decay_factor
                    log_train(log_fout, 'Decay learning rate to %f' %lr)
            elif cfgs.lr_update == 'step' and epoch_counter % cfgs.num_epochs_per_decay == 0:
                    lr = lr * cfgs.lr_decay_factor
                    log_train(log_fout, 'Decay learning rate to %f' %lr)
            if epoch_counter == FLAGS.n_epochs:
                if not do_snapshot:
                    log_train(log_fout, 'Final snapshotting')
                    model.save(FLAGS.output_dir)
                break
    log_fout.close()

if __name__ == '__main__':
    tf.app.run()
