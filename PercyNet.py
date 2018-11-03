#!/usr/bin/env python


import os
import argparse
import glob, time, six
import numpy as np
import skimage.io
import Augmentor
import shutil
from six.moves import range
from natsort import natsorted

import PIL
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tensorpack import *
from tensorpack.utils.gpu import get_num_gpu
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
import tensorflow as tf
from GAN import * #GANTrainer, GANModelDesc


SHAPE = 256
DIMY = 256
DIMX = 256
NF = 64  # channel size
Reduction = tf.losses.Reduction

def INReLU(x, name=None):
    x = InstanceNorm('inorm', x)
    return tf.nn.relu(x, name=name)


def INLReLU(x, name=None):
    x = InstanceNorm('inorm', x)
    return tf.nn.leaky_relu(x, alpha=0.2, name=name)

# Utility function for scaling 
def tf_2tanh(x, maxVal=255.0, name='ToRangeTanh'):
    with tf.variable_scope(name):
        return (x / maxVal - 0.5) * 2.0
###############################################################################
def tf_2imag(x, maxVal=255.0, name='ToRangeImag'):
    with tf.variable_scope(name):
        return (x / 2.0 + 0.5) * maxVal

# Utility function for scaling 
def np_2tanh(x, maxVal = 255.0, name='ToRangeTanh'):
    return (x / maxVal - 0.5) * 2.0
###############################################################################
def np_2imag(x, maxVal = 255.0, name='ToRangeImag'):
    return (x / 2.0 + 0.5) * maxVal


def time_seed ():
    seed = None
    while seed == None:
        cur_time = time.time ()
        seed = int ((cur_time - int (cur_time)) * 1000000)
    return seed

class ImageDataFlow(RNGDataFlow):
    def __init__(self, imageDir, paintDir, size, dtype='float32', isTrain=False, isValid=False, isTest=False):
        self.dtype      = dtype
        self.imageDir   = imageDir
        self.paintDir   = paintDir
        self._size      = size
        self.isTrain    = isTrain
        self.isValid    = isValid

        
        images = glob.glob(self.imageDir + '/*.jpg')
        paints = glob.glob(self.paintDir + '/*/*.jpg')

        OnlyAscii = lambda s: re.match('^[\x00-\x7F]+$', s) != None
        import re
        images = [st for st in images if OnlyAscii(st)]
        paints = [st for st in paints if OnlyAscii(st)]
        self.images = []
        self.paints = []
        self.data_seed = time_seed ()
        self.data_rand = np.random.RandomState(self.data_seed)
        self.rng = np.random.RandomState(999)
        # for i in range (len (images)):
        #     image = images[i]
        #     self.images.append (skimage.io.imread (image))
        # for i in range (len (paints)):
        #     paint = paints[i]
        #     self.paints.append (skimage.io.imread (paint))
        # for i in range (len (noises)):
        #     noise = noises[i]
        #     self.noises.append (skimage.io.imread (noise))
        self.images = images
        self.paints = paints

        print self.images, self.paints
    def size(self):
        return self._size

   


    ###############################################################################
    def get_data(self):
        for k in range(self._size):
            #
            # Pick randomly a tuple of training instance
            #
            isSmall = True
            while isSmall:                
                rand_image = self.data_rand.randint(0, len(self.images))
                rand_paint = self.data_rand.randint(0, len(self.paints))

                image = skimage.io.imread(self.images[rand_image])
                paint = skimage.io.imread(self.paints[rand_paint])

                if image.ndim==2:
                    image = np.stack((image, image, image), -1)
                if paint.ndim==2:
                    paint = np.stack((paint, paint, paint), -1)
                if image.shape[0] > DIMY and image.shape[1] > DIMX and paint.shape[0] > DIMY and paint.shape[1]> DIMX:
                    isSmall=False

            seed = time_seed () #self.rng.randint(0, 20152015)

            if self.isTrain:

                p = Augmentor.Pipeline()
                p.crop_centre(probability=1, percentage_area=0.99)
                #p.crop_by_size(probability=1, width=DIMX, height=DIMY, centre=True)
                p.resize(probability=1, width=DIMX, height=DIMY)
                p.zoom_random(probability=0.25, percentage_area=0.9)
                #p.flip_random(probability=0.75)
                p.flip_left_right(probability=0.5)

                image = p._execute_with_array(image) 

                p = Augmentor.Pipeline()
                p.crop_centre(probability=1, percentage_area=0.99)
                #p.crop_by_size(probability=1, width=DIMX, height=DIMY, centre=True)
                p.resize(probability=1, width=DIMX, height=DIMY)
                p.zoom_random(probability=0.25, percentage_area=0.9)
                #p.flip_random(probability=0.75)
                p.flip_left_right(probability=0.5)
                paint = p._execute_with_array(paint) 
            else:
                p = Augmentor.Pipeline()
                p.crop_centre(probability=1, percentage_area=0.99)
                #p.crop_by_size(probability=1, width=DIMX, height=DIMY, centre=True)
                p.resize(probability=1, width=DIMX, height=DIMY)
                image = p._execute_with_array(image) 
                
                p = Augmentor.Pipeline()
                p.crop_centre(probability=1, percentage_area=0.99)
                #p.crop_by_size(probability=1, width=DIMX, height=DIMY, centre=True)
                p.resize(probability=1, width=DIMX, height=DIMY)
                paint = p._execute_with_array(paint) 
                # pass
                

            #Expand dim to make single channel


            image = np.expand_dims(image, axis=0)
            paint = np.expand_dims(paint, axis=0)

            yield [image.astype(np.float32), 
                   paint.astype(np.float32), 
                   ] 

def get_data(dataDir, isTrain=True, isValid=False, isTest=False):
    # Process the directories 
    #if isTrain:
    num=500
    if isValid:
        num=1
    if isTest:
        num=1

    
    dset  = ImageDataFlow(os.path.join(dataDir, 'train2014'),
                          os.path.join(dataDir, 'wikiart'),
                          num, 
                          isTrain=isTrain, 
                          isValid=isValid, 
                          isTest =isTest)
    dset.reset_state()
    if isTrain:
        dset = PrefetchDataZMQ(dset, 32)
    return dset
    # dset1 = dataset.BSDS500(name='train', shuffle=True)
    # dset2 = dataset.BSDS500(name='train', shuffle=True)
    # if isTrain:
    #     shape_aug = [
    #         imgaug.RandomResize(xrange=(0.7, 1.5), yrange=(0.7, 1.5),
    #                             aspect_ratio_thres=0.15),
    #         imgaug.RotationAndCropValid(90),
    #         #CropMultiple16(),
    #         imgaug.Flip(horiz=True),
    #         imgaug.Flip(vert=True),
    #         imgaug.Resize((DIMY, DIMX))
    #     ]
    # else:
    #     # the original image shape (321x481) in BSDS is not a multiple of 16
    #     IMAGE_SHAPE = (DIMY, DIMX)
    #     shape_aug = [imgaug.CenterCrop(IMAGE_SHAPE)]
    # dset1 = AugmentImageComponents(dset1, shape_aug, (0, 1), copy=False)
    # dset2 = AugmentImageComponents(dset2, shape_aug, (0, 1), copy=False)

    # def f(m):   # thresholding
    #     m[m >= 0.50] = 1
    #     m[m < 0.50] = 0
    #     return m
    # dset1 = MapDataComponent(dset1, f, 1)
    # dset2 = MapDataComponent(dset2, f, 1)

    # if isTrain:
    #     augmentors = [
    #         imgaug.Brightness(63, clip=False),
    #         imgaug.Contrast((0.4, 1.5)),
    #     ]
    #     dset1 = AugmentImageComponent(dset1, augmentors, copy=False)
    #     dset2 = AugmentImageComponent(dset2, augmentors, copy=False)
    #     #dset1 = BatchDataByShape(dset1, 8, idx=0)
    #     #dset2 = BatchDataByShape(dset2, 8, idx=0)
    #     dset1 = PrefetchDataZMQ(dset1, 5)
    #     dset2 = PrefetchDataZMQ(dset2, 5)
    # else:
    #     #dset1 = BatchData(dset1, 1)
    #     #dset2 = BatchData(dset2, 1)
    #     dset1 = PrefetchDataZMQ(dset1, 1)
    #     dset2 = PrefetchDataZMQ(dset2, 1)
    #     #pass
    # # dset1 = MapData(dset1, lambda x: np.expand_dims(x, axis=0))
    # # dset2 = MapData(dset2, lambda x: np.expand_dims(x, axis=0))
    # # dset1 = MapDataComponent(dset1, lambda x: np.expand_dims(x, axis=-1), 1)
    # # dset2 = MapDataComponent(dset2, lambda x: np.expand_dims(x, axis=-1), 1)
    # # dset1 = MapDataComponent(dset1, lambda x: np.expand_dims(x, axis=-1), 3)
    # # dset2 = MapDataComponent(dset2, lambda x: np.expand_dims(x, axis=-1), 3)
    
    # return JoinData([dset1, dset2])
    # #dset = PrefetchDataZMQ(dset, 8 if isTrain else 1)
    # #return dset






def normalize(v):
    assert isinstance(v, tf.Tensor)
    v.get_shape().assert_has_rank(4)
    return v / tf.reduce_mean(v, axis=[1, 2, 3], keepdims=True)


def gram_matrix(v):
    assert isinstance(v, tf.Tensor)
    v.get_shape().assert_has_rank(4)
    dim = v.get_shape().as_list()
    v = tf.reshape(v, [-1, dim[1] * dim[2], dim[3]])
    return tf.matmul(v, v, transpose_a=True)


VGG19_MEAN_1 = np.array([123.68, 116.779, 103.939])  # RGB
VGG19_MEAN_TENSOR_1 = tf.constant(VGG19_MEAN_1, dtype=tf.float32)
VGG19_MEAN_2 = np.array([123.68, 116.779, 103.939, 123.68, 116.779, 103.939])  # RGB
VGG19_MEAN_TENSOR_2 = tf.constant(VGG19_MEAN_2, dtype=tf.float32)

@auto_reuse_variable_scope
def vgg19_encoder(x, name='VGG19_Encoder'):
    with argscope([Conv2D], kernel_shape=3, nl=INLReLU):
        conv1_1 = Conv2D('conv1_1', x, 64)
        conv1_2 = Conv2D('conv1_2', conv1_1, 64)
        pool1 = MaxPooling('pool1', conv1_2, 2)  # 64
        conv2_1 = Conv2D('conv2_1', pool1, 128)
        conv2_2 = Conv2D('conv2_2', conv2_1, 128)
        pool2 = MaxPooling('pool2', conv2_2, 2)  # 32
        conv3_1 = Conv2D('conv3_1', pool2, 256)
        conv3_2 = Conv2D('conv3_2', conv3_1, 256)
        conv3_3 = Conv2D('conv3_3', conv3_2, 256)
        conv3_4 = Conv2D('conv3_4', conv3_3, 256)
        pool3 = MaxPooling('pool3', conv3_4, 2)  # 16
        conv4_1 = Conv2D('conv4_1', pool3, 512)
        conv4_2 = Conv2D('conv4_2', conv4_1, 512)
        conv4_3 = Conv2D('conv4_3', conv4_2, 512)
        conv4_4 = Conv2D('conv4_4', conv4_3, 512)
        pool4 = MaxPooling('pool4', conv4_4, 2)  # 8
        conv5_1 = Conv2D('conv5_1', pool4, 512)
        conv5_2 = Conv2D('conv5_2', conv5_1, 512)
        conv5_3 = Conv2D('conv5_3', conv5_2, 512)
        conv5_4 = Conv2D('conv5_4', conv5_3, 512)
        pool5 = MaxPooling('pool5', conv5_4, 2)  # 4

        return pool5


def resnet_block(x, name, num_filters=512):
    with tf.variable_scope(name):
        y = Conv2D('conv0', x, num_filters, activation=tf.nn.relu)
        y = Conv2D('conv1', y, num_filters, activation=tf.identity)
    return x + y

def upsample(x, factor=2):
    _, h, w, _ = x.get_shape().as_list()
    x = tf.image.resize_nearest_neighbor(x, [factor * h, factor * w], align_corners=True)
    return x


@auto_reuse_variable_scope
def vgg19_decoder(x, last_dim=3, name='VGG19_Decoder'):
    with argscope([Conv2D], kernel_shape=3, nl=INLReLU):   
        # x = Conv2D('conv1', x, NF)
        # for i in range(10):
        #     x = resnet_block(x, 'block_%i' % i)
        x = upsample(x) # pool5
        x = Conv2D('conv_post_5_4', x, 512)
        x = Conv2D('conv_post_5_3', x, 512)
        x = Conv2D('conv_post_5_2', x, 512)
        x = Conv2D('conv_post_5_1', x, 512)
        x = upsample(x) # pool4
        x = Conv2D('conv_post_4_4', x, 512)
        x = Conv2D('conv_post_4_3', x, 512)
        x = Conv2D('conv_post_4_2', x, 512)
        x = Conv2D('conv_post_4_1', x, 512)
        x = upsample(x) # pool4
        x = Conv2D('conv_post_3_4', x, 256)
        x = Conv2D('conv_post_3_3', x, 256)
        x = Conv2D('conv_post_3_2', x, 256)
        x = Conv2D('conv_post_3_1', x, 256)
        x = upsample(x) # pool4
        x = Conv2D('conv_post_2_2', x, 128)
        x = Conv2D('conv_post_2_1', x, 128)
        x = upsample(x) # pool4
        x = Conv2D('conv_post_1_2', x, 64)
        x = Conv2D('conv_post_1_1', x, 64)        

        x = Conv2D('conv_post_0_0', x, last_dim, activation=tf.nn.tanh)
        return x
###############################################################################
class Model(GANModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, (None, DIMY, DIMX, 3), 'image'),
                tf.placeholder(tf.float32, (None, DIMY, DIMX, 3), 'paint'),
                ]

    @auto_reuse_variable_scope
    def encoder(self, x, last_dim=1):
        assert x is not None
        return vgg19_encoder(x)

    @auto_reuse_variable_scope
    def decoder(self, x, last_dim=3):
        assert x is not None
        return vgg19_decoder(x, last_dim=last_dim)

    @auto_reuse_variable_scope
    def generator(self, x, last_dim=3):
        assert x is not None
        x = x - VGG19_MEAN_TENSOR_2 / 255.0
        x = self.encoder(x)
        x = self.decoder(x, last_dim=last_dim)
        return x

    @auto_reuse_variable_scope
    def discriminator(self, x, last_dim=1):
        assert x is not None
        x = x - VGG19_MEAN_TENSOR_1 / 255.0
        x = self.encoder(x)
        return x

    def build_graph(self, I, S):

        with tf.variable_scope('gen'):
            IS = tf.concat([I, S], axis=-1)
            # IS = tf_2tanh(IS, maxVal=255.0)
            with tf.variable_scope('absorption'):
                C = self.generator(IS, last_dim=3)
                C   = tf_2imag(C, maxVal=255.0)
                C  = tf.identity(C, name='C')
            CS = tf.concat([C, S], axis=-1)
            # CS = tf_2tanh(CS, maxVal=255.0)
            with tf.variable_scope('separation'):
                Ir = self.generator(CS, last_dim=3)    
                Ir  = tf_2imag(Ir, maxVal=255.0)    
                Ir = tf.identity(Ir, name='Ir')

        ZZ = tf.zeros_like(I)

        with tf.variable_scope('discrim'):
            with tf.variable_scope('encoder'):
                ST_dis_real = self.discriminator(S)
                ST_dis_fake = self.discriminator(C) 

        



        viz = tf.concat([
                        tf.concat([I, S, C, Ir], axis=2), 
                        ], axis=1)
        viz = tf.clip_by_value(viz, 0, 255)
        viz = tf.cast(viz, tf.uint8, name='viz')
        print viz
        tf.summary.image('branch', viz, max_outputs=50)

        def LSGAN_losses(real, fake):
            d_real = tf.reduce_mean(tf.squared_difference(real, 1), name='d_real')
            d_fake = tf.reduce_mean(tf.square(fake), name='d_fake')
            d_loss = tf.multiply(d_real + d_fake, 0.5, name='d_loss')

            g_loss = tf.reduce_mean(tf.squared_difference(fake, 1), name='g_loss')
            add_moving_summary(g_loss, d_loss)
            return g_loss, d_loss

        g_losses = []
        d_losses = []
        ALPHA = 1e+2
        GAMMA = 1e-2
        LAMBDA = 1e-6

        def additional_losses(a, b, c):
            with tf.variable_scope('VGG19'):
                x = tf.concat([a, b, c], axis=0)
                x = x - VGG19_MEAN_TENSOR_1
                # VGG 19
                with varreplace.freeze_variables():
                    with argscope(Conv2D, kernel_size=3, activation=tf.nn.relu):
                        conv1_1 = Conv2D('conv1_1', x, 64)
                        conv1_2 = Conv2D('conv1_2', conv1_1, 64)
                        pool1 = MaxPooling('pool1', conv1_2, 2)  # 64
                        conv2_1 = Conv2D('conv2_1', pool1, 128)
                        conv2_2 = Conv2D('conv2_2', conv2_1, 128)
                        pool2 = MaxPooling('pool2', conv2_2, 2)  # 32
                        conv3_1 = Conv2D('conv3_1', pool2, 256)
                        conv3_2 = Conv2D('conv3_2', conv3_1, 256)
                        conv3_3 = Conv2D('conv3_3', conv3_2, 256)
                        conv3_4 = Conv2D('conv3_4', conv3_3, 256)
                        pool3 = MaxPooling('pool3', conv3_4, 2)  # 16
                        conv4_1 = Conv2D('conv4_1', pool3, 512)
                        conv4_2 = Conv2D('conv4_2', conv4_1, 512)
                        conv4_3 = Conv2D('conv4_3', conv4_2, 512)
                        conv4_4 = Conv2D('conv4_4', conv4_3, 512)
                        pool4 = MaxPooling('pool4', conv4_4, 2)  # 8
                        conv5_1 = Conv2D('conv5_1', pool4, 512)
                        conv5_2 = Conv2D('conv5_2', conv5_1, 512)
                        conv5_3 = Conv2D('conv5_3', conv5_2, 512)
                        conv5_4 = Conv2D('conv5_4', conv5_3, 512)
                        pool5 = MaxPooling('pool5', conv5_4, 2)  # 4

            # perceptual loss
            with tf.name_scope('perceptual_loss'):
                pool2 = normalize(pool2)
                pool5 = normalize(pool5)
                phi_a_1, phi_b_1, _ = tf.split(pool2, 3, axis=0)
                phi_a_2, phi_b_2, _ = tf.split(pool5, 3, axis=0)

                logger.info('Create perceptual loss for layer {} with shape {}'.format(pool2.name, pool2.get_shape()))
                pool2_loss = tf.losses.mean_squared_error(phi_a_1, phi_b_1, reduction=Reduction.MEAN)
                logger.info('Create perceptual loss for layer {} with shape {}'.format(pool5.name, pool5.get_shape()))
                pool5_loss = tf.losses.mean_squared_error(phi_a_2, phi_b_2, reduction=Reduction.MEAN)

            # texture loss
            with tf.name_scope('texture_loss'):
                def texture_loss(x, p=16):
                    _, h, w, c = x.get_shape().as_list()
                    x = normalize(x)
                    assert h % p == 0 and w % p == 0
                    logger.info('Create texture loss for layer {} with shape {}'.format(x.name, x.get_shape()))

                    x = tf.space_to_batch_nd(x, [p, p], [[0, 0], [0, 0]])  # [b * ?, h/p, w/p, c]
                    x = tf.reshape(x, [p, p, -1, h // p, w // p, c])       # [p, p, b, h/p, w/p, c]
                    x = tf.transpose(x, [2, 3, 4, 0, 1, 5])                # [b * ?, p, p, c]
                    _, patches_a, patches_b = tf.split(x, 3, axis=0)          # each is b,h/p,w/p,p,p,c

                    patches_a = tf.reshape(patches_a, [-1, p, p, c])       # [b * ?, p, p, c]
                    patches_b = tf.reshape(patches_b, [-1, p, p, c])       # [b * ?, p, p, c]
                    return tf.losses.mean_squared_error(
                        gram_matrix(patches_a),
                        gram_matrix(patches_b),
                        reduction=Reduction.MEAN
                    )

                texture_loss_conv1_1 = tf.identity(texture_loss(conv1_1), name='normalized_conv1_1')
                texture_loss_conv2_1 = tf.identity(texture_loss(conv2_1), name='normalized_conv2_1')
                texture_loss_conv3_1 = tf.identity(texture_loss(conv3_1), name='normalized_conv3_1')

            return [pool2_loss, pool5_loss, texture_loss_conv1_1, texture_loss_conv2_1, texture_loss_conv3_1]

        with tf.name_scope('losses'):
            additional_losses = additional_losses(I, C, S)
            with tf.name_scope('content'):
                g_losses.append(tf.multiply(2e-1, additional_losses[0], name="loss_LP1"))
                g_losses.append(tf.multiply(2e-2, additional_losses[1], name="loss_LP2"))
            
            with tf.name_scope('style'):
                g_losses.append(tf.multiply(3e-7, additional_losses[2], name="loss_LT1"))
                g_losses.append(tf.multiply(1e-6, additional_losses[3], name="loss_LT2"))
                g_losses.append(tf.multiply(1e-6, additional_losses[4], name="loss_LT3"))
            
            with tf.name_scope('gan'):
                # gan loss
                G_loss, D_loss = self.build_losses(ST_dis_real, ST_dis_fake)
                g_losses.append(G_loss)
                d_losses.append(D_loss)

            with tf.name_scope('recon'):
                recon_loss_I = tf.reduce_mean(tf.abs(I - Ir), name='recon_loss_I')
                g_losses.append(GAMMA*recon_loss_I)
            
                

        self.g_loss = tf.reduce_sum(g_losses, name='G_loss_total')
        self.d_loss = tf.reduce_sum(d_losses, name='D_loss_total')
            

        self.collect_variables('gen', 'discrim')
        add_moving_summary(recon_loss_I,
                           # tv_loss,
                           additional_losses[0],
                           additional_losses[1],
                           additional_losses[2],
                           additional_losses[3],
                           additional_losses[4],
                           self.g_loss, 
                           self.d_loss)
        

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=2e-5, trainable=False)
        return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)


class VisualizeTestSet(Callback):
    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['image', 'paint'], ['viz'])

    def _before_train(self):
        global args
        self.val_ds = get_data(args.data, isTrain=False, isValid=True)
        self.val_ds.reset_state()

    def _trigger(self):
        idx = 0
        for lst in self.val_ds.get_data():
            viz = self.pred(lst)
            self.trainer.monitors.put_image('test_viz_{}'.format(idx), viz)
            idx += 1



'''
MIT License
Copyright (c) 2018 Fanjin Zeng
This work is licensed under the terms of the MIT license, see <https://opensource.org/licenses/MIT>.  
'''

def sliding_window_view(x, shape, step=None, subok=False, writeable=False):
    """
    Create sliding window views of the N dimensions array with the given window
    shape. Window slides across each dimension of `x` and provides subsets of `x`
    at any window position.
    Parameters
    ----------
    x : ndarray
        Array to create sliding window views.
    shape : sequence of int
        The shape of the window. Must have same length as number of input array dimensions.
    step: sequence of int, optional
        The steps of window shifts for each dimension on input array at a time.
        If given, must have same length as number of input array dimensions.
        Defaults to 1 on all dimensions.
    subok : bool, optional
        If True, then sub-classes will be passed-through, otherwise the returned
        array will be forced to be a base-class array (default).
    writeable : bool, optional
        If set to False, the returned array will always be readonly view.
        Otherwise it will return writable copies(see Notes).
    Returns
    -------
    view : ndarray
        Sliding window views (or copies) of `x`. view.shape = (x.shape - shape) // step + 1
    See also
    --------
    as_strided: Create a view into the array with the given shape and strides.
    broadcast_to: broadcast an array to a given shape.
    Notes
    -----
    ``sliding_window_view`` create sliding window views of the N dimensions array
    with the given window shape and its implementation based on ``as_strided``.
    Please note that if writeable set to False, the return is views, not copies
    of array. In this case, write operations could be unpredictable, so the return
    views is readonly. Bear in mind, return copies (writeable=True), could possibly
    take memory multiple amount of origin array, due to overlapping windows.
    For some cases, there may be more efficient approaches, such as FFT based algo discussed in #7753.
    Examples
    --------
    >>> i, j = np.ogrid[:3,:4]
    >>> x = 10*i + j
    >>> shape = (2,2)
    >>> sliding_window_view(x, shape)
    array([[[[ 0,  1],
             [10, 11]],
            [[ 1,  2],
             [11, 12]],
            [[ 2,  3],
             [12, 13]]],
           [[[10, 11],
             [20, 21]],
            [[11, 12],
             [21, 22]],
            [[12, 13],
             [22, 23]]]])
    >>> i, j = np.ogrid[:3,:4]
    >>> x = 10*i + j
    >>> shape = (2,2)
    >>> step = (1,2)
    >>> sliding_window_view(x, shape, step)
    array([[[[ 0,  1],
             [10, 11]],
            [[ 2,  3],
             [12, 13]]],
           [[[10, 11],
             [20, 21]],
            [[12, 13],
             [22, 23]]]])
    """
    # first convert input to array, possibly keeping subclass
    x = np.array(x, copy=False, subok=subok)

    try:
        shape = np.array(shape, np.int)
    except:
        raise TypeError('`shape` must be a sequence of integer')
    else:
        if shape.ndim > 1:
            raise ValueError('`shape` must be one-dimensional sequence of integer')
        if len(x.shape) != len(shape):
            raise ValueError("`shape` length doesn't match with input array dimensions")
        if np.any(shape <= 0):
            raise ValueError('`shape` cannot contain non-positive value')

    if step is None:
        step = np.ones(len(x.shape), np.intp)
    else:
        try:
            step = np.array(step, np.intp)
        except:
            raise TypeError('`step` must be a sequence of integer')
        else:
            if step.ndim > 1:
                raise ValueError('`step` must be one-dimensional sequence of integer')
            if len(x.shape)!= len(step):
                raise ValueError("`step` length doesn't match with input array dimensions")
            if np.any(step <= 0):
                raise ValueError('`step` cannot contain non-positive value')

    o = (np.array(x.shape)  - shape) // step + 1 # output shape
    if np.any(o <= 0):
        raise ValueError('window shape cannot larger than input array shape')

    strides = x.strides
    view_strides = strides * step

    view_shape = np.concatenate((o, shape), axis=0)
    view_strides = np.concatenate((view_strides, strides), axis=0)
    #view = np.lib.stride_tricks.as_strided(x, view_shape, view_strides, subok=subok, writeable=writeable)
    view = np.lib.stride_tricks.as_strided(x, view_shape, view_strides, subok=subok)#, writeable=writeable)

    if writeable:
        return view.copy()
    else:
        return view


def sample(dataDir, model_path, prefix='.'):
    print("Starting...")
    print(dataDir)
    imageFiles = glob.glob(os.path.join(dataDir, '*.tif'))
    print(imageFiles)
    # Load the model 
    predict_func = OfflinePredictor(PredictConfig(
        model=Model(),
        session_init=get_model_loader(model_path),
        input_names=['image'],
        output_names=['fA']))

    for imageFile in imageFiles:
        head, tail = os.path.split(imageFile)
        print tail
        dstFile = prefix+tail
        print dstFile

        # Read the image file
        image = skimage.io.imread(imageFile)

        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        

        print(image.shape)
        def weighted_map_blocks(arr, inner, outer, ghost, func=None): # work for 3D, inner=[1, 3, 3], ghost=[0, 2, 2], 
            dtype = np.float32 #arr.dtype

            arr = arr.astype(np.float32)
            # param
            if outer==None:
                outer = inner + 2*ghost
                outer = [(i + 2*g) for i, g in zip(inner, ghost)]
            shape = outer
            steps = inner
                
            print(outer)
            print(shape)
            print(inner)
            
            padding=arr.copy()
            print(padding.shape)
            #print(padding)
            
            weights = np.zeros_like(padding)
            results = np.zeros_like(padding)
            
            v_padding = sliding_window_view(padding, shape, steps)
            v_weights = sliding_window_view(weights, shape, steps)
            v_results = sliding_window_view(results, shape, steps)
            
            print 'v_padding', v_padding.shape


            #for z in range(v_padding.shape[0]):
            for y in range(v_padding.shape[1]):
                for x in range(v_padding.shape[2]):
    
                    v_result = np.array(func(
                                            (v_padding[0,y,x,0][...,0:1]) ) ) ### Todo function is here
                    v_result = np.squeeze(v_result, axis=0).astype(np.float32)
    
                    yy, xx = np.meshgrid(np.linspace(-1,1,shape[1], dtype=np.float32), 
                                         np.linspace(-1,1,shape[2], dtype=np.float32))
                    d = np.sqrt(xx*xx+yy*yy)
                    sigma, mu = 0.5, 0.0
                    v_weight = 1e-6+np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
                    v_weight = v_weight/v_weight.max()
                    #print v_weight.shape
                    #v_weight.tofile('gaussian_map.npy')
                    
                    v_weight = np.expand_dims(v_weight, axis=-1)
                    #print shape
                    #print v_weight.shape
                    v_weights[0,y,x] += v_weight

                    v_results[0,y,x] += v_result * v_weight
                        
            # Divided by the weight param
            results /= weights 
            
            
            return results.astype(dtype)
    

        dst = weighted_map_blocks(image, inner=[1, 256, 256, 1], 
                                           outer=[1, 512, 512, 1], 
                                           ghost=[1, 256, 256, 0], 
                                           func=predict_func) # inner,  ghost

        dst = np.squeeze(dst)
        skimage.io.imsave(dstFile, np_2imag(dst, maxVal=255.0).astype(np.uint8))
    return None




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', required=True,
        help='the image directory. should contain trainA/trainB/testA/testB')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--gpu',        default='0', help='comma seperated list of GPU(s) to use.')
    parser.add_argument('--vgg19',      help='load model',         default='data/model/vgg19.npz') #vgg19.npz')
    parser.add_argument('--sample',     default='', help='deploy')
    
    args = parser.parse_args()

    # Set the GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.sample:
        # TODO
        print("Deploy the data")
        prefix="result/"
        if os.path.exists(prefix):
            shutil.rmtree(prefix, ignore_errors=True)
        os.makedirs(prefix) 
        sample(args.data, args.load, prefix = prefix)
        # pass
    else:   
        logger.auto_set_dir()

        if args.load: 
            session_init=SaverRestore(args.load)
        else: 
            assert os.path.isfile(args.vgg19)

            weight = dict(np.load(args.vgg19))
            param_dict = {}
            param_dict.update({'gen/encoder/' + name: value for name, value in six.iteritems(weight)})
            param_dict.update({'discrim/encoder/' + name: value for name, value in six.iteritems(weight)})
            param_dict.update({'VGG19/' + name: value for name, value in six.iteritems(weight)})
            session_init = DictRestore(param_dict)
            df = get_data(args.data)
            df = PrintData(df)
            #data = StagingInput(QueueInput(df))
            data = QueueInput(df)

            nr_tower = max(get_num_gpu(), 1)
            if nr_tower == 1:
                trainer = GANTrainer(data, Model())
            else:
                trainer = MultiGPUGANTrainer(nr_tower, data, Model())
            trainer.train_with_defaults(
                callbacks=[
                    PeriodicTrigger(ModelSaver(), every_k_epochs=500),
                    ScheduledHyperParamSetter(
                        'learning_rate',
                        [(100, 2e-5), (2000000, 0)], interp='linear'),
                    #PeriodicTrigger(VisualizeTestSet(), every_k_epochs=1),
                ],
                max_epoch=2000000,
                steps_per_epoch=data.size(),
                session_init=session_init #SaverRestore(args.load) if args.load else None
            )
