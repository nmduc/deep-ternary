import os
import cv2 as cv
import tensorflow as tf
import numpy as np
import configs.configs as configs
from models.autoencoder import Autoencoder
import math 

FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string("snapshot_dir", "./outputs/snapshots/", "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_string("model_fname", "", "Name of the pretrained model checkpoints (to resume from)")
tf.flags.DEFINE_string("test_folder", "./test_images", "")
tf.flags.DEFINE_string("reconstruction_folder", "./reconstructions/", "")
tf.flags.DEFINE_string("prefix", "", "")
tf.flags.DEFINE_string("inference_mode", "overlap", "")
tf.flags.DEFINE_integer("stride", 2, "")
cfgs = configs.CONFIGS

def list_image(folder):
    ### List all images (jpg, png, tif) inside the given folder 
    files = os.listdir(folder)
    img_list = []
    for filename in files:
	filename = filename.strip()
        flag = filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.tif')
        if not flag:
            continue
        img_list.append(filename)
    return img_list

def load_image(fname):
    img = cv.imread(fname)
    return img

def toGrayscale(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return gray

def overlap_inference(model, img, bs, stride=8):
    w, sbw, sc = model.get_sensing_matrix()
    assert np.array_equal(np.unique(sbw), np.array([-1.,0.,1.])), 'Sparse binary sensing matrix must contains only -1, 0, 1'
    B = cfgs.patch_size
    assert B % stride == 0
    big_img = cv.copyMakeBorder(img, B, B, B, B, cv.BORDER_REFLECT_101)
    recon_img = np.zeros(big_img.shape)
    occurence_map = np.zeros(big_img.shape, dtype=np.int32)
    H, W = big_img.shape[0], big_img.shape[1]
    ite = 0
    n_patches = (int(H/stride) + 1) * (int(W/stride) + 1)
    patches = np.zeros((n_patches, B*B), dtype=np.float32)
    for y in xrange(0, H, stride):
        for x in xrange(0, W, stride):
            if y + B >= H or x + B >= W:
                continue
            patch = big_img[y:y+B,x:x+B]
            patch = patch.reshape(-1)
            patches[ite,:] = patch
            occurence_map[y:y+B,x:x+B] += 1
            ite += 1
    recons = np.zeros_like(patches)
    start = 0
    while True:
        end = start + bs
        if start + bs > n_patches:
            end = n_patches
        # sense first to get the measurement
        measurement = np.multiply(np.matmul(patches[start:end,:], sbw), sc)
        # reconstruct patch from measurement
        recons[start:end,:] = model.reconstruct(measurement)
        if end == n_patches:
            break
        start = end
    ite = 0
    for y in xrange(0, H, stride):
        for x in xrange(0, W, stride):
            if y + B >= H or x + B >= W:
                continue
            recon_img[y:y+B,x:x+B] += recons[ite,:].reshape(B,B)
            ite += 1
    occurence_map[occurence_map==0] = 1
    recon_img /= occurence_map
    recon_img = recon_img[B:-B,B:-B]
    return recon_img

def psnr(ref, img):
    assert ref.shape == img.shape, 'Image shapes mis-match'
    mse = np.sum(np.square(ref - img)) / ref.size
    MAX_I = 255.0
    psnr = 20 * math.log(MAX_I, 10) - 10 * math.log(mse, 10)
    return psnr

def main(unused_argv):
    # load test images
    test_list = list_image(FLAGS.test_folder)
    # load model 
    assert (FLAGS.snapshot_dir != "" or FLAGS.model_fname != ""), 'No pretrained model specified'
    model = Autoencoder(cfgs.patch_size*cfgs.patch_size, cfgs, log_dir=None)
    snapshot_fname = FLAGS.model_fname if FLAGS.model_fname != "" \
        else tf.train.latest_checkpoint(FLAGS.snapshot_dir)
    model.restore(snapshot_fname)
    print('Restored from %s' %snapshot_fname)
    sum_psnr = 0.0
    stride = FLAGS.stride
    for img_fname in test_list:
        orig_img = load_image('%s/%s' %(FLAGS.test_folder,img_fname))
        # pre-process image
        gray_img = toGrayscale(orig_img)
        img = gray_img.astype(np.float32)
        img -= cfgs.mean_value
        img *= cfgs.scale
        # make measurement and reconstruct image
        recon_img = overlap_inference(model, img, bs=cfgs.batch_size, stride=stride)
        recon_img /= cfgs.scale
        recon_img += cfgs.mean_value
        # save reconstruction 
        cv.imwrite('%s/%sOI_%d_%s' %(FLAGS.reconstruction_folder, FLAGS.prefix, stride, img_fname), recon_img.astype(np.uint8))
        psnr_ = psnr(gray_img.astype(np.float32), recon_img)
        print('Image %s, psnr: %f' %(img_fname, psnr_))
        sum_psnr += psnr_
    mean_psnr = sum_psnr / len(test_list)

    print('---------------------------')
    print('Mean PSNR: %f' %mean_psnr)

if __name__ == '__main__':
    tf.app.run()
