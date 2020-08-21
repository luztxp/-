# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import logging
import pickle
import sys
from time import localtime, strftime
from net_input_everything_featparts4 import *
from ops import *
from utils import *
relu = tf.nn.relu
import tensorflow as tf
import os
import heapq
from sklearn.decomposition import PCA
import Test1 as T
import cell as C
import os.path
import time
import numpy as np
import shutil
import face_recognition
import os

# X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])  #导入数据，维度为4
# pca = PCA(n_components=2)   #降到2维
# pca.fit(X)                  #训练
# newX=pca.fit_transform(X)   #降维后的数据

# 设置 GPU 按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
os.system('echo $PATH')
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  #指定前二块GPU可用
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # doesn’t enable AVX/FMA
#These parameters should provide a good initialization, but if for specific refinement, you can adjust them during training.

# SYM_W = 0.3  #lamana1
ALPHA_ADVER = 1e-2 #lamana2
BELTA_FEATURE = 1e-3 ##lamana3
# FEATURE = 1e-3
TV_WEIGHT = 1e-5 #lamana4
# IDEN_W = 1e-3  #a er fa

UPDATE_G = 1  #optimize D once and UPDATE_G times G
UPDATE_D = 1

MODE = 'fs60'  #'f' feature loss enabled.   'v' -verification enanbled. 'o' original, 'm' masked is mandatory and no need to specify
LOAD_60_LABEL = False #otherwise load frontal label
WITHOUT_CODEMAP = True
USE_MASK = False
RANDOM_VERIFY = False
CHANNEL = 3

###########################################################

############################################################
flags = tf.app.flags
flags.DEFINE_integer("epoch", 150, "Epoch to train [250]")
flags.DEFINE_float("learning_rate", 2e-4, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
#flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 1, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 128, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("output_size", 128, "The size of the output images to produce [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("dataset", "MultiPIE", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "data1/check2", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "data1/samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", True, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS
#tf.subtract(x, y, name=None)   # 减法
class DCGAN(object):
    def __init__(self, sess,batch_size=10,output_size=128, gf_dim=64, df_dim=64,
                 dataset_name='MultiPIE',checkpoint_dir=None):
        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [64]
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.save_interval = 1242   #621
        self.sample_interval = 1242 #1242
        self.flage = False

        self.testing = True  # False  True
        self.sess = sess
        self.batch_size = 6
        self.time_size = 2
        self.lstm = True

        self.test_batch_size = self.batch_size   ##  5
        self.output_size = output_size           ## 128

        self.gf_dim = gf_dim #眼睛使用的内核深度:64     ##生成器中第一个卷积层的filter维度
        self.df_dim = df_dim #???64                     ##鉴别器中第一个卷积层的filter维度

        self.z_dim = 100

        random.seed()
        self.DeepFacePath = '/media/gpu/文档/lsf/TP-GAN/DeepFace168.pickle'
        self.dataset_name = dataset_name #MultiPIE
        self.checkpoint_dir = checkpoint_dir  #None
        self.loadDeepFace(self.DeepFacePath)

    def build_model(self):
        #lambda语句中，冒号前是参数，可以有多个，用逗号隔开，冒号右边的返回值。lambda语句构建的其实是一个函数对象。
        #例：m = lambda x,y,z: (x-y)*z;   n = m(234,122,5)  #则n=560
        #python中的[::-1]是将涉及的数字倒序输出，即将图片反转

        # mc = lambda left : tf.concat([left, left[:,:,::-1,:]], 3)
        self.images_with_code2 = tf.placeholder(tf.float32, [self.batch_size] + [self.time_size] + [self.output_size, self.output_size, CHANNEL], name='images_with_code')
        self.labels = tf.placeholder(tf.float32, [self.batch_size] + [self.output_size, self.output_size, CHANNEL], name='label_images')
        self.idenlabels = tf.placeholder(tf.int32, [self.batch_size]) #int32 (10,)

        self.g32_labels = tf.image.resize_bilinear(self.labels, [32, 32]) #双线性插值法 float32 (10,32,32,3)
        self.g64_labels = tf.image.resize_bilinear(self.labels, [64, 64]) #双线性插值法 float32 (10,64,64,3)

        self.images_with_code = tf.split(self.images_with_code2,self.time_size,1)
        self.feats_lstms = list()
        self.images_with_codes = list()
        self.fts_loss = list()


        for i in range(self.time_size):
            self.feats_lstm = list()
            self.images_with_code[i] = tf.reshape(self.images_with_code[i],[self.images_with_code[i].shape[0],self.images_with_code[i].shape[2],self.images_with_code[i].shape[3],self.images_with_code[i].shape[4]])
            if i == 0:
                reuse = False
            else:
                reuse = True
            self.feats = self.generator(self.images_with_code[i], self.batch_size, name="encoder",reuse=reuse)

            self.feats = list(self.feats)
            for j in range(len(self.feats)):
                # if j == len(self.feats)-1:
                #     feats_00 = self.feats[j]
                #     feats_00 = tf.reshape(feats_00,[feats_00.shape[0],1,1,feats_00.shape[1]])
                #     self.feats[j] = feats_00
                feats_0 = tf.reshape(self.feats[j],[self.feats[j].shape[0],1,self.feats[j].shape[1],self.feats[j].shape[2],self.feats[j].shape[3]])
                if i == 0:
                    self.feats_lstm.append(feats_0)
                if i > 0:
                    feats_1 = tf.concat([feats_0,self.feats_lstms[i-1][j]],1)
                    self.feats_lstm.append(feats_1)
            self.feats_lstms.append(self.feats_lstm)

        # self.z = tf.random_normal([self.batch_size, self.z_dim], mean=0.0, stddev=0.02, seed=2017)  # (10,100)正太分布
        self.vector_1 = self.generator(self.labels, self.batch_size, name="encoder", reuse=True)
        # bb = self.FeatureExtractDeepFace(tf.image.rgb_to_grayscale(self.labels), reuse=True)
        # self.vector_1 = bb[0], bb[1], bb[2], bb[3]

        if self.lstm:
            h_state_0 = self.network2(self.feats_lstms[-1][0], "network2_h0")
            h_state_1 = self.network2(self.feats_lstms[-1][1], "network2_h1")
            h_state_2 = self.network2(self.feats_lstms[-1][2], "network2_h2")
            h_state_3 = self.network2(self.feats_lstms[-1][3], "network2_h3")  #(6,16,16,256)
            # h_state_4 = self.network2(self.feats_lstms[-1][4], "network2_h4")  #(6,512)
            self.h_state_5 = self.network3(h_state_0,h_state_1,h_state_2,h_state_3, "network3_h5")
            self.vector = self.vector_1
        else:
            for i in range(self.time_size):
                h_state_00 = self.feats_lstms[-1][0][:, i, :, :, :]
                h_state_11 = self.feats_lstms[-1][1][:, i, :, :, :]
                h_state_22 = self.feats_lstms[-1][2][:, i, :, :, :]
                h_state_33 = self.feats_lstms[-1][3][:, i, :, :, :]
                # h_state_44 = self.feats_lstms[-1][4][:, i, :, :, :]
                # h_state_44 = tf.reshape(h_state_44,[h_state_44.shape[0],h_state_44.shape[-1]])
                self.vector = self.vector_1
                if i == 0:
                    h_state_0 = h_state_00
                    h_state_1 = h_state_11
                    h_state_2 = h_state_22
                    h_state_3 = h_state_33
                    # h_state_4 = h_state_44
                else:
                    h_state_0 = tf.concat([h_state_0, h_state_00], -1)
                    h_state_1 = tf.concat([h_state_1, h_state_11], -1)
                    h_state_2 = tf.concat([h_state_2, h_state_22], -1)
                    h_state_3 = tf.concat([h_state_3, h_state_33], -1)
                    # h_state_4 = tf.concat([h_state_4, h_state_44], -1)
                    # self.h_state_5 = self.network3(h_state_0, h_state_1, h_state_2, h_state_3, "network3_h5")

                    self.vector = list(self.vector_1)
                    self.vector[0] = tf.concat([self.vector[0], self.vector_1[0]], -1)
                    self.vector[1] = tf.concat([self.vector[1], self.vector_1[1]], -1)
                    self.vector[2] = tf.concat([self.vector[2], self.vector_1[2]], -1)
                    self.vector[3] = tf.concat([self.vector[3], self.vector_1[3]], -1)
                    # self.vector[4] = tf.concat([self.vector[4], self.vector_1[4]], -1)
            self.h_state_5 = self.network3(h_state_0, h_state_1, h_state_2, h_state_3, "network3_h5")

        self.feats_lstm = h_state_0, h_state_1, h_state_2, h_state_3, self.h_state_5  # , self.profiles

        self.G, self.G2, self.G3, self.G4_fus = \
            self.decoder(*self.feats_lstm, batch_size=self.batch_size)
        self.img128 = self.G, self.G2, self.G3, self.G4_fus

        # self.z_labels = tf.random_normal([self.batch_size, 128,128,3], mean=0.0, stddev=0.02, seed=2017)  # (10,100)从服从指定正太分布的数值中取出指定个数的值。
        # self.vector += (self.z_labels,)
        # self.G_label, self.G2_label, self.G3_label, self.G4_label = self.decoder(*self.vector, batch_size=self.batch_size, reuse=True)

        # # self.ErrL = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.abs(self.vector[0] - h_state_0),  1), 1),1))
        # self.ErrL = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.abs(self.vector[1] - h_state_1), 1), 1),1))
        # self.ErrL += tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.abs(self.vector[2] - h_state_2), 1), 1),1))
        # self.ErrL += tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.abs(self.vector[3] - h_state_3), 1), 1),1))
        # self.ErrL += tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.abs(self.vector[4] - h_state_5), 1), 1),1))
        self.ErrL = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.abs(self.labels - self.h_state_5), 1), 1),1))

        print("Using local discriminator!")
        self.D, self.D_logits = self.discriminatorLocal(self.labels)  # input(10,128,128,3) output(10,16),(10,16)
        self.D_, self.D_logits_ = self.discriminatorLocal(self.G,reuse=True)  # input(10,128,128,3) output(10,16),(10,16)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits,labels=tf.ones_like(self.D) * 0.9))  ##计算经sigmoid 函数激活之后的交叉熵 input(10,16)
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_adver = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))

        # self.d_loss_real = self.D
        # self.d_loss_fake = self.D_
        # self.d_loss = tf.ones_like(self.D) - self.D + self.D_

        a, b, c, d, self.G_pool5, self.Gvector = self.FeatureExtractDeepFace(tf.image.rgb_to_grayscale(self.G))  # reduce_mean()在某一维度求平均值
        label_a, label_b, label_c, label_d, self.label_pool5, self.labelvector = self.FeatureExtractDeepFace(tf.image.rgb_to_grayscale(self.labels), reuse=True)
        self.dv_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(self.Gvector - self.labelvector), 1))
        self.dv_loss += tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.abs(self.G_pool5 - self.label_pool5), 1), 1),1))

        # _, self.identitylogits_label, _ = self.FeaturePredict(self.vector[-1],reuse=True)
        # self.iden_labels = tf.subtract(self.idenlabels, 1, name=None)
        # self.idenloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.iden_labels,logits=self.identitylogits))  # 计算logits和labels之间的稀疏softmax交叉熵
        # self.idenloss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.iden_labels,logits=self.identitylogits_label))  # 计算logits和labels之间的稀疏softmax交叉熵

        errL1 = tf.abs(self.G - self.labels)  #(6,128,128,3)
        errL1_2 = tf.abs(self.G2 - self.g64_labels)  # * mask64
        errL1_3 = tf.abs(self.G3 - self.g32_labels)  # * mask32

        self.weightedErrL1 = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(errL1, 1), 1), 1))
        self.weightedErrL2 = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(errL1_2, 1), 1), 1))
        self.weightedErrL3 = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(errL1_3, 1), 1), 1))

        # errL1_label = tf.abs(self.G_label - self.labels)  # (6,128,128,3)
        # errL1_2_label = tf.abs(self.G2_label - self.g64_labels)  # * mask64
        # errL1_3_label = tf.abs(self.G3_label - self.g32_labels)  # * mask32
        #
        # self.weightedErrL1 += 0.001 * tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(errL1_label, 1), 1), 1))
        # self.weightedErrL2 += 0.001 * tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(errL1_2_label, 1), 1), 1))
        # self.weightedErrL3 += 0.001 * tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(errL1_3_label, 1), 1), 1))

        self.tv_loss = tf.reduce_mean(total_variation(self.G))  # 计算全变分,用于图像降噪

        self.g_loss = self.weightedErrL1 + self.weightedErrL2 + self.weightedErrL3 + self.ErrL  #+ FEATURE * self.ErrL

        self.g_loss += BELTA_FEATURE * self.dv_loss + self.tv_loss * TV_WEIGHT + ALPHA_ADVER * self.g_loss_adver# + IDEN_W * self.idenloss
        self.g_loss = self.g_loss*0.01

        self.var_file = open('var_log.txt', mode='a')
        t_vars = [var for var in tf.trainable_variables() if 'FeatureExtractDeepFace' not in var.name \
                             and 'processor' not in var.name]#tf.trainable_variables返回需要训练的变量列表

        self.d_vars = [var for var in t_vars if 'discriminatorLocal' in var.name]
        # self.all_g_vars = [var for var in t_vars if 'discriminatorLocal' not in var.name]
        self.dec_vars = [var for var in t_vars if 'decoder' in var.name and 'select' not in var.name]
        self.enc_vars = [var for var in t_vars if 'encoder' in var.name]
        self.pre_vars = [var for var in t_vars if 'FeaturePredict' in var.name]
        self.lstm2_vars = [var for var in t_vars if 'network2' in var.name]
        self.lstm3_vars = [var for var in t_vars if 'network3' in var.name]

        self.ed_vars = list(self.dec_vars); self.ed_vars.extend(self.enc_vars);
        self.ed_vars.extend(self.pre_vars); self.ed_vars.extend(self.lstm2_vars);
        self.ed_vars.extend(self.lstm3_vars);

        self.saver = tf.train.Saver(t_vars, max_to_keep=2) #创建saver对象，用于向文件夹中写入包含了当前模型中所有可训练变量的 checkpoint 文件

    def train(self, config):
        """Train DCGAN"""
        data = MultiPIE(LOAD_60_LABEL=LOAD_60_LABEL, GENERATE_MASK=USE_MASK, RANDOM_VERIFY=RANDOM_VERIFY, MIRROR_TO_ONE_SIDE = True)#(False,False,False,True,'FS')
        #random.randint()用于生成一个指定范围内的整数,(1,100000) 在sample_dir字符串的后面直接加上（1,1000）的随机数
        config.sample_dir += '/'.format(random.randint(1, 1000))

        start_time_d_optim = time.time();   #
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)  #Adam优化算法
        print("d_optim costs %s seconds" % (time.time() - start_time_d_optim))

        start_time_g_dec_optim = time.time();
        optimizer = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1)
        g_dec_optim = optimizer.minimize(self.g_loss, var_list=self.ed_vars)
        # g_dec_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
        #                    .minimize(self.g_loss, var_list=self.ed_vars)
        print("g_dec_optim costs %s seconds" % (time.time() - start_time_g_dec_optim))

        start_time_init_op = time.time();
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        print("init_op costs %s seconds" % (time.time() - start_time_init_op))

        counter = 0

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:  #print("[*] Reading checkpoints...")    print(" [*] Failed to find a checkpoint")
            print(" [!] Load failed...")

        if not self.testing: #训练阶段    # self.testing = False
            print("start training!")
            self.gs_loss = [99999999999999]
            self.ds_loss = [99999999999999]

            True_rats = []
            for epoch in range(config.epoch): #全部样本个数epoch=250
                self.g_loss_1 = []
                self.d_loss_1 =[]
                # epoch = epoch + 2
                ##双斜杠//，先做除法（/）,然后向下取整（floor）。至少有一方是float型时，结果为float型；两个数都是int型时，结果为int型。
                if self.flage:
                    break
                batch_idxs = min(data.size, config.train_size) // self.batch_size//self.time_size
                for idx in range(0, batch_idxs):
                    poses = np.empty([self.batch_size,self.time_size])
                    filenames1 = np.empty([self.batch_size,self.time_size])
                    batch_iden1 = np.empty([self.batch_size, self.time_size])
                    batch_labels1 = np.empty([self.batch_size,self.time_size,128,128,3])
                    batch_images_with_code1 = np.empty([self.batch_size, self.time_size, 128, 128, 3])
                    for jdx in range(self.time_size):
                        batch_images_with_code, batch_labels, pose,batch_iden,filenames,labelnames,batch_profile\
                                     = data.next_image_and_label_mask_batch(self.batch_size,self.time_size, idx, jdx)
                        batch_images_with_code1[:,jdx,...] = batch_images_with_code
                        batch_labels1[:,jdx,...] = batch_labels
                        batch_iden1[:,jdx] = batch_iden

                    batch_images_with_code = batch_images_with_code1
                    batch_labels = batch_labels1[:,-1,...]
                    batch_iden = batch_iden1[:,-1]

                    counter += 1

                    start_time_run_d_optim = time.time()
                    for dd in range(2):
                        err_d = self.sess.run([d_optim,self.d_loss_real, self.d_loss_fake, self.d_loss],
                                 feed_dict={self.images_with_code2: batch_images_with_code,
                                            self.labels: batch_labels})
                    self.d_loss_1.append(err_d[-1])
                    # print('counter = %s,' % counter, 'epoch = %s,' % epoch, "fake =  %s" % err_d[-2],'real  = %s,' % err_d[-3])
                    print("sess.run(d_optim) costs %s seconds" % (time.time() - start_time_run_d_optim),'err_d = %s,' % err_d[-1])

                    start_time_run_rot_optim = time.time()
                    err_g=self.sess.run([g_dec_optim, self.g_loss],
                        feed_dict={self.images_with_code2: batch_images_with_code,
                                   self.labels: batch_labels,
                                   self.idenlabels: batch_iden})
                    print("sess.run(g_optim) costs %s seconds" % (time.time() - start_time_run_rot_optim),'err_g = %s,' % err_g[-1])
                    self.g_loss_1.append(err_g[-1])

                    print('counter = %s,' % counter, 'epoch = %s,' % epoch)

                    # if np.mod(counter, self.save_interval) == self.save_interval-1:
                    if idx == batch_idxs - 1:
                        self.ds_loss.append(np.mean(self.d_loss_1))
                        self.gs_loss.append(np.mean(self.g_loss_1))
                        print(self.gs_loss)
                        print(self.ds_loss)
                        self.save(config.checkpoint_dir, counter) #保存模型的训练参数

                    # if np.mod(counter, self.sample_interval) > self.sample_interval-21 and np.mod(counter, self.sample_interval) <= self.sample_interval-1: #mod()求余数运算

                        samples = self.sess.run(
                                self.img128,
                                feed_dict={self.images_with_code2: batch_images_with_code,
                                            })
                        for i in range(self.time_size):
                            batch_images_with_code1 = batch_images_with_code[:,i,...]
                            savedtest = save_images(batch_images_with_code1 if WITHOUT_CODEMAP else batch_images_with_code1[...,0:3], [100, 100],
                                                './{}/{:02d}/{:04d}_{}_'.format(config.sample_dir, epoch, idx, i), suffix='')
                        savedoutput = save_images(batch_labels, [128, 128],
                                                  './{}/{:02d}/{:04d}_{}_'.format(config.sample_dir, epoch, idx, i),suffix='_label')
                        savedoutput = save_images(samples[0], [128, 128],
                                                  './{}/{:02d}/{:04d}_{}_'.format(config.sample_dir, epoch, idx, i),suffix='_128')
                        savedoutput = save_images(samples[1], [64, 64],
                                                  './{}/{:02d}/{:04d}_{}_'.format(config.sample_dir, epoch, idx, i),suffix='_64')
                        # savedoutput = save_images(samples[2], [32, 32],
                        #                           './{}/{:02d}/{:04d}_{}_'.format(config.sample_dir, epoch, idx, i),suffix='_32')
                        savedoutput = save_images(samples[3], [128, 128],
                                                  './{}/{:02d}/{:04d}_{}_'.format(config.sample_dir, epoch, idx, i),suffix='_fus')
                        # savedoutput = save_images(samples_label, [128, 128],
                        #                            './{}/{:02d}/{:04d}_{}'.format(config.sample_dir, epoch, idx, i), suffix='_label_syn')

                        batch_idxsx = data.test_size // self.batch_size // self.time_size
                        synfinall_sample_dir = './data1/222/'
                        for idxx in range(0, batch_idxsx):
                            sample_images1 = np.empty([self.batch_size, self.time_size, 128, 128, 3])
                            filenames1 = []
                            for jdxx in range(self.time_size):
                                sample_images, filenames, _, _, _, _ = data.test_batch(self.batch_size, self.time_size,idxx, jdxx)
                                sample_images1[:, jdxx, ...] = sample_images
                                filenames1.append(filenames)
                            sample_images = sample_images1

                            syn_names1 = list()
                            samples = self.sess.run(
                                self.img128,
                                feed_dict={self.images_with_code2: sample_images,
                                           })

                            for k in range(self.time_size):
                                syn_names = list()
                                for i in range(self.batch_size):
                                    if k == 0:
                                        syn_name = filenames1[k][i][0:7] + filenames1[k][i][10:13]  # + ''.join('syn.png')
                                        syn_names.append(syn_name)
                                        if i == self.batch_size - 1:
                                            syn_names1.append(syn_names)
                                    if k > 0:
                                        syn_name = syn_names1[k - 1][i] + '_' + filenames1[k][i][10:13]  # +  ''.join('syn.png')
                                        syn_names.append(syn_name)
                                        if i == self.batch_size - 1:
                                            syn_names1.append(syn_names)
                            syn_filename = syn_names1[-1]
                            savedoutput = save_images(samples[0], [128, 128], synfinall_sample_dir + '/',filelist=syn_filename)

                        syn_sample_dir = synfinall_sample_dir
                        # syn_90_dir = synfinall_sample_dir+ '/' + 'syn_90°/'
                        # filenames_syn = os.listdir(syn_sample_dir)
                        # for syn_filename in filenames_syn:
                        #     if syn_filename[7:14] == '110_240' or syn_filename[7:14] == '240_110':
                        #         shutil.copy(syn_sample_dir + syn_filename, syn_90_dir + syn_filename)

                        # syn_90_dir = synfinall_sample_dir
                        filepath = '/media/gpu/文档/lsf/TP-GAN/data1/syn/test_label/'  # 识别标签 即原图
                        filenames = sorted(os.listdir(filepath))
                        filenames2 = sorted(os.listdir(syn_sample_dir))
                        class_num = len(filenames2)
                        count = 0
                        # 依次将合成的图片与原图进行比较识别
                        for name2 in filenames2:  # 待识别样本 即合成图像
                            diss = []
                            unknown_picture = face_recognition.load_image_file(syn_sample_dir + name2)
                            a = face_recognition.face_encodings(unknown_picture)

                            if len(a) > 0 :
                                unknown_face_encoding = a[0]
                                if len(a) > 1 :
                                    print('a length = %s,' % len(a))
                            else:
                                break
                            # unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]
                            for i in range(len(filenames)):  # 识别标签 即原图
                                picture_of_me = face_recognition.load_image_file(filepath + filenames[i])
                                my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]
                                result, dis = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)
                                diss.append(dis[0])
                            min_m = [i for (i, v) in enumerate(diss) if v == min(diss)]
                            # min_m = list(map(diss.index, heapq.nsmallest(1, diss)))
                            for j in range(len(min_m)):
                                if filenames[min_m[j]][0:3] == name2[0:3]:  # [12:15]  [7:10]
                                    count = count + 1
                        print('count = %s,' % count,'class_num = %s,' % class_num)
                        True_rat = count / class_num
                        True_rats.append(True_rat)
                        print('True_rats = %s,' % True_rats, 'ok')
                        print('当前测试集90°识别率 = %s,' % True_rats[-1])
                        # if len(True_rats) > 1 and True_rats[-1] < True_rats[-2] and True_rats[-2] and True_rats[-1]>0.84:
                        if True_rats[-1] > 0.9: #and self.gs_loss[-1] > self.gs_loss[-2]:
                            self.flage = True
                            break
                        tf.gfile.DeleteRecursively(syn_sample_dir)  # 删除该目录下的全部内容
                        tf.gfile.MkDir(syn_sample_dir)    #重新创建文件夹

        else:
            batch_idxs = data.test_size // self.batch_size // self.time_size
            count_totall = count_15 = count_30 = count_45 = count_60 = count_75 = count_90 = 0
            synfinall_sample_dir = './data1/222/'
            for idx in range(0, batch_idxs):
                sample_images1 = np.empty([self.batch_size, self.time_size, 128, 128, 3])
                filenames1 = []
                for jdx in range(self.time_size):
                    sample_images, filenames, _, _, _, _ = data.test_batch(self.batch_size, self.time_size, idx, jdx)
                    sample_images1[:,jdx,...] = sample_images
                    filenames1.append(filenames)

                sample_images = sample_images1
                print('test samples reading complete{}'.format(idx))

                syn_names1 = list()
                samples = self.sess.run(
                    self.img128,
                    feed_dict={self.images_with_code2: sample_images,
                               })
                for k in range(self.time_size):
                    syn_names = list()
                    for i in range(self.batch_size):
                        if k == 0:
                            syn_name = filenames1[k][i][0:7] + filenames1[k][i][10:13]  # + ''.join('syn.png')
                            syn_names.append(syn_name)
                            if i == self.batch_size - 1:
                                syn_names1.append(syn_names)
                        if k > 0:
                            syn_name = syn_names1[k - 1][i] + '_' + filenames1[k][i][10:13]  # +  ''.join('syn.png')
                            syn_names.append(syn_name)
                            if i == self.batch_size - 1:
                                syn_names1.append(syn_names)
                syn_filename = syn_names1[-1]
                savedoutput = save_images(samples[0], [128, 128],synfinall_sample_dir + '/', filelist=syn_filename)

            syn_sample_dir = synfinall_sample_dir
            filepath = '/media/gpu/文档/lsf/TP-GAN/data1/syn/test_label_01/'  # 识别标签 即原图
            # filepath = './data1/MultiPIE/totall_test_0°/'
            filenames = sorted(os.listdir(filepath))
            filenames2 = sorted(os.listdir(syn_sample_dir))
            class_num = len(filenames2)
            count = 0
            # 依次将合成的图片与原图进行比较识别
            for name2 in filenames2:  # 待识别样本 即合成图像
                diss = []
                unknown_picture = face_recognition.load_image_file(syn_sample_dir + name2)
                a = face_recognition.face_encodings(unknown_picture)
                if len(a) > 0:
                    unknown_face_encoding = a[0]
                    if len(a) > 1:
                        print('a length = %s,' % len(a))
                else:
                    break
                for i in range(len(filenames)):  # 识别标签 即原图
                    picture_of_me = face_recognition.load_image_file(filepath + filenames[i])
                    my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]
                    result, dis = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)
                    diss.append(dis[0])
                min_m = [i for (i, v) in enumerate(diss) if v == min(diss)]
                # min_m = list(map(diss.index, heapq.nsmallest(1, diss)))
                for j in range(len(min_m)):
                    if filenames[min_m[j]][0:3] == name2[0:3]:  # [12:15]  [7:10]
                        count = count + 1

            num_single = (data.test_size // self.time_size) // 6
            print('count = %s,' % count, 'class_num = %s,' % class_num, 'num_single = %s,' % num_single)

            print('当前测试集识别率 = {}'.format(count / class_num))
            # print('rate_15 = {}'.format(count_15 / num_single))
            # print('rate_30 = {}'.format(count_30 / num_single))
            # print('rate_45 = {}'.format(count_45 / num_single))
            # print('rate_60 = {}'.format(count_60 / num_single))
            # print('rate_75 = {}'.format(count_75 / num_single))
            # print('rate_90 = {}'.format(count_90 / num_single))

    def processor(self, images, reuse=False):
        'accept 3 channel images, output orginal 3 channels and 3 x 4 gradient map-> 15 channels'
        with tf.variable_scope("processor") as scope:
            if reuse:
                scope.reuse_variables()
            input_dim = images.get_shape()[-1]
            gradientKernel = gradientweight()
            output_dim = gradientKernel.shape[-1]
            print("processor:output_dim equals ", output_dim)
            k_hw = gradientKernel.shape[0]
            init = tf.constant_initializer(value=gradientKernel, dtype=tf.float32)
            w = tf.get_variable('w', [k_hw, k_hw, input_dim, output_dim],
                                initializer=init)
            conv = tf.nn.conv2d(images, w, strides=[1, 1, 1, 1], padding='SAME')
            #conv = conv * 2
            return tf.concat([images, conv], 3)

    def FeaturePredict(self, featvec, reuse=False):
        with tf.variable_scope("FeaturePredict") as scope:
            if reuse:
                scope.reuse_variables()
            c4r4_l = tf.reshape(featvec,[self.batch_size, -1])
            c7_l = linear(c4r4_l, output_size=1024,scope='feature', bias_start=0.1, with_w=True)[0]
            # c7_l_m = linear(c7_l, output_size=512, scope='feature_2', bias_start=0.1, with_w=True)[0]
            identitylogits = linear(Dropout(c7_l, keep_prob=0.8, is_training= not self.testing), output_size=200, scope='idenLinear', bias_start=0.1, with_w=True)[0]
            return None, identitylogits, None

    def discriminatorLocal(self, images, reuse=False):
        with tf.variable_scope("discriminatorLocal") as scope:
            if reuse:
                scope.reuse_variables()
            h0 = lrelu(conv2d(images, self.df_dim//2, name='d_h0_conv'))
            #64
            h1 = lrelu(batch_norm(conv2d(h0, self.df_dim, name='d_h1_conv'), name='d_bn1'))
            #32
            h2 = lrelu(batch_norm(conv2d(h1, self.df_dim*2, name='d_h2_conv'), name='d_bn2'))
            #16
            h3 = lrelu(batch_norm(conv2d(h2, self.df_dim*4, name='d_h3_conv'), name='d_bn3'))
            # #8x8x512
            h3r1 = resblock(h3, name = "d_h3_conv_res1")
            h4 = lrelu(batch_norm(conv2d(h3r1, self.df_dim*8, name='d_h4_conv'), name='d_bn4'))
            h4r1 = resblock(h4, name = "d_h4_conv_res1")

            h5 = lrelu(batch_norm(conv2d(h4r1, 1024, d_h=4, d_w=4, name='d_h5_conv'), name='d_bn5'))
            h5 = tf.reshape(h5, [self.batch_size, 1024])
            h6 = tf.layers.dense(inputs=h5, units=1, activation=lrelu)
            h6 = tf.reshape(h6, [self.batch_size])

            # c4r4_l = tf.reshape(h4r1, [self.batch_size, -1])
            # h5 = linear(c4r4_l, output_size=1024, scope='feature_1', bias_start=0.1, with_w=True)[0]
            # h6 = linear(h5, output_size=1, scope='feature', bias_start=0.1, with_w=True)[0]

            return h6, h6

    def decoder(self, feat128, feat64, feat32, feat16, feat128_fus,
                batch_size = 10, name="decoder", reuse = False):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            # initial_all = featvec
            # before_select16 = resblock(feat16, name="select16_res_1")
            initial_32 = relu(deconv2d(feat16, [batch_size, 32, 32, self.gf_dim], name="initial32"))
            initial_64 = relu(deconv2d(initial_32, [batch_size, 64, 64, self.gf_dim//2], name="initial64"))
            initial_128 = relu(deconv2d(initial_64, [batch_size, 128, 128, self.gf_dim // 4], name="initial128"))

            before_select32 = resblock(tf.concat([feat32, initial_32], 3), name="select32_res_1")
            # reconstruct32 = resblock(resblock(tf.concat([reconstruct32_deconv, before_select32], 3), name="dec32_res1"), name="dec32_res2")
            reconstruct32 = resblock(before_select32, name="dec32_res1")
            img32 = tf.nn.sigmoid(conv2d(reconstruct32, 3, d_h=1, d_w=1, name="check_img32"))
            reconstruct64_deconv = relu(batch_norm(deconv2d(reconstruct32, [batch_size, 64, 64, self.gf_dim*2], name="g_deconv64"), name="g_bnd3"))

            before_select64 = resblock(tf.concat([feat64, initial_64], 3), k_h=3, k_w=3,name="select64_res_1")

            reconstruct64 = resblock(resblock(tf.concat([reconstruct64_deconv, before_select64,
                                                tf.image.resize_bilinear(img32, [64,64])], 3), name="dec64_res1"), name="dec64_res2")
            img64 = tf.nn.sigmoid(conv2d(reconstruct64, 3, d_h=1, d_w=1, name="check_img64"))
            reconstruct128_deconv = relu(batch_norm(deconv2d(reconstruct64, [batch_size, 128, 128, self.gf_dim], name="g_deconv128"), name="g_bnd4"))

            before_select128 = resblock(tf.concat([feat128, initial_128,feat128_fus], 3), k_h=3, k_w=3,name="select128_res_1")

            reconstruct128 = resblock(tf.concat([reconstruct128_deconv, before_select128,
                                                 tf.image.resize_bilinear(img64, [128, 128])], 3),  k_h=3, k_w=3, name="dec128_res1")

            reconstruct128_1 = lrelu(batch_norm(conv2d(reconstruct128, self.gf_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="recon128_conv"), name="recon128_bnc"))
            # reconstruct128_1_r = resblock(reconstruct128_1, name="dec128_res2")
            reconstruct128_2 = lrelu(batch_norm(conv2d(reconstruct128_1, self.gf_dim/2, d_h=1, d_w=1, name="recon128_conv2"),name="recon128_bnc2"))
            img128 = tf.nn.sigmoid(conv2d(reconstruct128_2, 3, d_h=1, d_w=1, name="check_img128"))
            return img128, img64, img32, feat128_fus

    def generator(self, images, batch_size, name='generator', reuse = False):
        '''
        U-Net structure, slightly different from the original on the location of relu/lrelu
        :param images: IMAGE_SIZE x IMAGE_SIZE x CHANNEL
        :param batch_size:
        :param name:
        :param reuse:
        :return: labels: IMAGE_SIZE x IMAGE_SIZE x 3
        '''
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            #128x128x64
            c0 = lrelu(conv2d(images, self.gf_dim//2, d_h=1, d_w=1, name="g_conv0"))
            c0r = resblock(c0, k_h=3, k_w=3, name="g_conv0_res")
            c1 = lrelu(batch_norm(conv2d(c0r, self.gf_dim, name="g_conv1"), name="g_bnc1"))
            #64x64x64
            c1r = resblock(c1, k_h=3, k_w=3, name="g_conv1_res")
            c2 = lrelu(batch_norm(conv2d(c1r, self.gf_dim*2, name='g_conv2'),name="g_bnc2"))
            #32x32x128
            c2r = resblock(c2, name="g_conv2_res")
            c3 = lrelu(batch_norm(conv2d(c2r, self.gf_dim*4, name='g_conv3'),name="g_bnc3"))
            # # 16x16x256
            c3r = resblock(c3, name="g_conv3_res")
            c3r2 = resblock(c3r, name="g_conv3_res2")
            # c4 = lrelu(batch_norm(conv2d(c3r, self.gf_dim*4, name='g_conv4'),name="g_bnc4"))
            # # # 8x8x512
            # c4r = resblock(c4, name="g_conv4_res")
            # c4r2 = resblock(c4r, name="g_conv4_res2")
            # c4r3 = resblock(c4r2, name="g_conv4_res3")
            # c4r4 = resblock(c4r3, name="g_conv4_res4")
            # c4r4_l = tf.reshape(c4r4,[batch_size, -1])
            #
            # c7_l = linear(c4r4_l, output_size=1024,scope='feature', bias_start=0.1, with_w=True)[0]
            # c7_l_m = linear(c7_l, output_size=512, scope='feature_2', bias_start=0.1, with_w=True)[0]
            # # c7_l_m = tf.maximum(c7_l[:, 0:256], c7_l[:, 256:]) #取最大的一边(左/右)
            return c0r, c1r, c2r, c3r2

    def network1(self, inputs, batch_size, name, reuse=False):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            print(inputs.shape)
            if self.lstm:
                if len(inputs.shape) < 4:
                    inputs = tf.reshape(inputs,[inputs.shape[0],1,1,inputs.shape[1]])
                # cell = T.ConvLSTMCell(conv_ndims=2, input_shape=[inputs.shape[2], inputs.shape[3], inputs.shape[4]],
                #                       output_channels=inputs.shape[4], kernel_shape=[3, 3])

                # cell = C.ConvGRUCell([inputs.shape[2], inputs.shape[3]], inputs.shape[-1], [3, 3])
                cell = C.ConvLSTMCell([inputs.shape[2], inputs.shape[3]], inputs.shape[4], [3, 3])

                # outputs, h_state = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype, initial_state=cell.zero_state(batch_size=batch_size, dtype=tf.float32))  #, initial_state=initial_state
                outputs, h_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32,
                    initial_state=cell.zero_state(batch_size=batch_size, dtype=tf.float32), time_major=False)
                h_state = h_state.h

                if h_state.shape[2] == 1:
                    h_state = tf.reshape(h_state, [h_state.shape[0], h_state.shape[3]])

                return h_state

    def network2(self, inputs, name, reuse=False):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            print(inputs.shape)
            # lstm_forward_1 = C.ConvLSTMCell([inputs.shape[2], inputs.shape[3]], inputs.shape[4], [3, 3])
            # lstm_backward_1 = C.ConvLSTMCell([inputs.shape[2], inputs.shape[3]], inputs.shape[4], [3, 3])
            lstm_forward_1 = T.ConvLSTMCell(conv_ndims=2, input_shape=[inputs.shape[2], inputs.shape[3], inputs.shape[4]],output_channels=inputs.shape[4], kernel_shape=[3, 3])
            lstm_backward_1 = T.ConvLSTMCell(conv_ndims=2,input_shape=[inputs.shape[2], inputs.shape[3], inputs.shape[4]],output_channels=inputs.shape[4], kernel_shape=[3, 3])

            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_forward_1, cell_bw=lstm_backward_1, inputs=inputs, dtype=tf.float32, time_major=False)
            fw = states[0].h
            bw = states[1].h

            feat_128 = resblock(tf.concat([fw, bw], 3), name="g_conv1_res")
            feat_128 = lrelu(batch_norm(conv2d(feat_128,inputs.shape[-1], d_h=1, d_w=1, name="fusion_128"),name="fus_128"))

            if feat_128.shape[2] == 1:
                feat_128 = tf.reshape(feat_128, [feat_128.shape[0], feat_128.shape[3]])
            return feat_128

    def network3(self, feat_128, feat_64, feat_32, feat_16,
               name="network3",reuse = False):
         with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            feats_128_deconv = relu(batch_norm(conv2d(feat_128, 3, d_h=1, d_w=1, name="conv_128"), name="f_bnd0"))
            feats_64_deconv = relu(batch_norm(deconv2d(feat_64, [self.batch_size, 128, 128, 3], name="f3_deconv64"), name="f_bnd3"))
            feats_32_deconv = relu(batch_norm(deconv2d(feat_32, [self.batch_size, 128, 128, 3], d_h=4, d_w=4, name="f2_deconv32"), name="f_bnd2"))
            feats_16_deconv = relu(batch_norm(deconv2d(feat_16, [self.batch_size, 128, 128, 3], d_h=8, d_w=8, name="f1_deconv16"), name="f_bnd1"))

            if self.lstm:
                feats_128_deconv = tf.reshape(feats_128_deconv,[feats_128_deconv.shape[0],1,feats_128_deconv.shape[1],feats_128_deconv.shape[2],feats_128_deconv.shape[3]])
                feats_64_deconv = tf.reshape(feats_64_deconv, [feats_64_deconv.shape[0], 1, feats_64_deconv.shape[1], feats_64_deconv.shape[2], feats_64_deconv.shape[3]])
                feats_32_deconv = tf.reshape(feats_32_deconv, [feats_32_deconv.shape[0], 1, feats_32_deconv.shape[1], feats_32_deconv.shape[2], feats_32_deconv.shape[3]])
                feats_16_deconv = tf.reshape(feats_16_deconv, [feats_16_deconv.shape[0], 1, feats_16_deconv.shape[1], feats_16_deconv.shape[2], feats_16_deconv.shape[3]])

                feat_128 = tf.concat([feats_128_deconv,feats_64_deconv,feats_32_deconv,feats_16_deconv],1)
                # lstm_forward_1 = C.ConvLSTMCell([feat_128.shape[2], feat_128.shape[3]], feat_128.shape[4], [3, 3])
                # lstm_backward_1 = C.ConvLSTMCell([feat_128.shape[2], feat_128.shape[3]], feat_128.shape[4], [3, 3])
                lstm_forward_1 = T.ConvLSTMCell(conv_ndims=2, input_shape=[feat_128.shape[2], feat_128.shape[3],feat_128.shape[4]], output_channels=feat_128.shape[4], kernel_shape=[3, 3])
                lstm_backward_1 = T.ConvLSTMCell(conv_ndims=2, input_shape=[feat_128.shape[2], feat_128.shape[3],feat_128.shape[4]], output_channels=feat_128.shape[4], kernel_shape=[3, 3])
                outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_forward_1, cell_bw=lstm_backward_1,inputs=feat_128, dtype=tf.float32, time_major=False)
                fw = states[0].h
                bw = states[1].h

                feat_128_fus = resblock(tf.concat([fw, bw], 3), name="g_conv1_res")
                feat_128_fus = tf.nn.sigmoid(conv2d(feat_128_fus, 3, d_h=1, d_w=1, name="fus_img128"))
            else:
                feat_128_fus = tf.concat([feats_128_deconv,feats_64_deconv,feats_32_deconv,feats_16_deconv], 3)
                feat_128_fus = resblock(feat_128_fus, name="g_conv1_res")
                feat_128_fus = tf.nn.sigmoid(conv2d(feat_128_fus, 3, d_h=1, d_w=1, name="fus_img128"))
            return feat_128_fus

    def loadDeepFace(self, DeepFacePath):
        if DeepFacePath is None:
            path = sys.modules[self.__class__.__module__].__file__
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "DeepFace.pickle")
            DeepFacePath = path
            logging.info("Load npy file from '%s'.", DeepFacePath)
        if not os.path.isfile(DeepFacePath):
            logging.error(("File '%s' not found. "), DeepFacePath)
            sys.exit(1)
        with open(DeepFacePath,'rb') as file:
          self.data_dict = pickle.load(file, encoding="iso-8859-1")
          # self.data_dict = pickle.load(file)
        print("Deep Face pickle data file loaded")

    def FeatureExtractDeepFace(self, images, P=False, name = "FeatureExtractDeepFace", reuse=False):
        'Preprocessing: from color to gray(reduce_mean)'
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            conv1 = self._conv_layer(images, name='conv1')
            slice1_1, slice1_2 = tf.split(conv1, 2, 3)
            eltwise1 = tf.maximum(slice1_1, slice1_2)

            slice11_1, slice11_2, slice11_3 = tf.split(conv1, 3, 3)
            eltwise11 = tf.maximum(tf.maximum(slice11_1, slice11_2), slice11_3)
            # slice11_1, slice11_2, slice11_3 = tf.split(eltwise1, 3, 3)
            # eltwise11_1 = tf.maximum(tf.maximum(slice11_1, slice11_2), slice11_3)
            # eltwise11 = tf.concat([eltwise11_1, eltwise1],3)

            pool1 = tf.nn.max_pool(eltwise1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],#池化层用于缩小矩阵尺寸，从而减少最后全连接层中的参数，还可以防止过拟合
                                   padding='SAME')
            conv2_1 = self._conv_layer(pool1, name='conv2_1')
            slice2_1_1, slice2_1_2 = tf.split(conv2_1, 2, 3)
            eltwise2_1 = tf.maximum(slice2_1_1, slice2_1_2)

            conv2_2 = self._conv_layer(eltwise2_1, name='conv2_2')
            slice2_2_1, slice2_2_2 = tf.split(conv2_2, 2, 3)
            eltwise2_2 = tf.maximum(slice2_2_1, slice2_2_2)

            res2_1 = pool1 + eltwise2_2

            conv2a = self._conv_layer(res2_1, name='conv2a')
            slice2a_1, slice2a_2 = tf.split(conv2a, 2, 3)
            eltwise2a = tf.maximum(slice2a_1, slice2a_2)

            conv2 = self._conv_layer(eltwise2a, name='conv2')
            slice2_1, slice2_2 = tf.split(conv2, 2, 3)
            eltwise2 = tf.maximum(slice2_1, slice2_2)

            slice22_1, slice22_2, slice22_3 = tf.split(conv2, 3, 3)
            eltwise22 = tf.maximum(tf.maximum(slice22_1, slice22_2), slice22_3)

            pool2 = tf.nn.max_pool(eltwise2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME')

            conv3_1 = self._conv_layer(pool2, name='conv3_1')
            slice3_1_1, slice3_1_2 = tf.split(conv3_1, 2, 3)
            eltwise3_1 = tf.maximum(slice3_1_1, slice3_1_2)

            conv3_2 = self._conv_layer(eltwise3_1, name='conv3_2')
            slice3_2_1, slice3_2_2 = tf.split(conv3_2, 2, 3)
            eltwise3_2 = tf.maximum(slice3_2_1, slice3_2_2)

            res3_1 = pool2 + eltwise3_2

            conv3_3 = self._conv_layer(res3_1, name='conv3_3')
            slice3_3_1, slice3_3_2 = tf.split(conv3_3, 2, 3)
            eltwise3_3 = tf.maximum(slice3_3_1, slice3_3_2)

            conv3_4 = self._conv_layer(eltwise3_3, name='conv3_4')
            slice3_4_1, slice3_4_2 = tf.split(conv3_4, 2, 3)
            eltwise3_4 = tf.maximum(slice3_4_1, slice3_4_2)

            res3_2 = res3_1 + eltwise3_4

            conv3a = self._conv_layer(res3_2, name='conv3a')
            slice3a_1, slice3a_2 = tf.split(conv3a, 2, 3)
            eltwise3a = tf.maximum(slice3a_1, slice3a_2)

            conv3 = self._conv_layer(eltwise3a, name='conv3')
            slice3_1, slice3_2 = tf.split(conv3, 2, 3)
            eltwise3 = tf.maximum(slice3_1, slice3_2)

            slice33_1, slice33_2, slice33_3 = tf.split(conv3, 3, 3)
            eltwise33 = tf.maximum(tf.maximum(slice33_1, slice33_2),slice33_3)

            pool3 = tf.nn.max_pool(eltwise3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME')

            conv4_1 = self._conv_layer(pool3, name='conv4_1')
            slice4_1_1, slice4_1_2 = tf.split(conv4_1, 2, 3)
            eltwise4_1 = tf.maximum(slice4_1_1, slice4_1_2)

            conv4_2 = self._conv_layer(eltwise4_1, name='conv4_2')
            slice4_2_1, slice4_2_2 = tf.split(conv4_2, 2, 3)
            eltwise4_2 = tf.maximum(slice4_2_1, slice4_2_2)

            res4_1 = pool3 + eltwise4_2

            conv4_3 = self._conv_layer(res4_1, name='conv4_3')
            slice4_3_1, slice4_3_2 = tf.split(conv4_3, 2, 3)
            eltwise4_3 = tf.maximum(slice4_3_1, slice4_3_2)

            conv4_4 = self._conv_layer(eltwise4_3, name='conv4_4')
            slice4_4_1, slice4_4_2 = tf.split(conv4_4, 2, 3)
            eltwise4_4 = tf.maximum(slice4_4_1, slice4_4_2)

            res4_2 = res4_1 + eltwise4_4

            conv4_5 = self._conv_layer(res4_2, name='conv4_5')
            slice4_5_1, slice4_5_2 = tf.split(conv4_5, 2, 3)
            eltwise4_5 = tf.maximum(slice4_5_1, slice4_5_2)

            conv4_6 = self._conv_layer(eltwise4_5, name='conv4_6')
            slice4_6_1, slice4_6_2 = tf.split(conv4_6, 2, 3)
            eltwise4_6 = tf.maximum(slice4_6_1, slice4_6_2)

            res4_3 = res4_2 + eltwise4_6

            conv4a = self._conv_layer(res4_3, name='conv4a')
            slice4a_1, slice4a_2 = tf.split(conv4a, 2, 3)
            eltwise4a = tf.maximum(slice4a_1, slice4a_2)

            conv4 = self._conv_layer(eltwise4a, name='conv4')
            slice4_1, slice4_2 = tf.split(conv4, 2, 3)
            eltwise4 = tf.maximum(slice4_1, slice4_2)

            eltwise44 = conv4

            conv5_1 = self._conv_layer(eltwise4, name='conv5_1')
            slice5_1_1, slice5_1_2 = tf.split(conv5_1, 2, 3)
            eltwise5_1 = tf.maximum(slice5_1_1, slice5_1_2)

            conv5_2 = self._conv_layer(eltwise5_1, name='conv5_2')
            slice5_2_1, slice5_2_2 = tf.split(conv5_2, 2, 3)
            eltwise5_2 = tf.maximum(slice5_2_1, slice5_2_2)

            res5_1 = eltwise4 + eltwise5_2

            conv5_3 = self._conv_layer(res5_1, name='conv5_3')
            slice5_3_1, slice5_3_2 = tf.split(conv5_3, 2, 3)
            eltwise5_3 = tf.maximum(slice5_3_1, slice5_3_2)

            conv5_4 = self._conv_layer(eltwise5_3, name='conv5_4')
            slice5_4_1, slice5_4_2 = tf.split(conv5_4, 2, 3)
            eltwise5_4 = tf.maximum(slice5_4_1, slice5_4_2)

            res5_2 = res5_1 + eltwise5_4

            conv5_5 = self._conv_layer(res5_2, name='conv5_5')
            slice5_5_1, slice5_5_2 = tf.split(conv5_5, 2, 3)
            eltwise5_5 = tf.maximum(slice5_5_1, slice5_5_2)

            conv5_6 = self._conv_layer(eltwise5_5, name='conv5_6')
            slice5_6_1, slice5_6_2 = tf.split(conv5_6, 2, 3)
            eltwise5_6 = tf.maximum(slice5_6_1, slice5_6_2)

            res5_3 = res5_2 + eltwise5_6

            conv5_7 = self._conv_layer(res5_3, name='conv5_7')
            slice5_7_1, slice5_7_2 = tf.split(conv5_7, 2, 3)
            eltwise5_7 = tf.maximum(slice5_7_1, slice5_7_2)

            conv5_8 = self._conv_layer(eltwise5_7, name='conv5_8')
            slice5_8_1, slice5_8_2 = tf.split(conv5_8, 2, 3)
            eltwise5_8 = tf.maximum(slice5_8_1, slice5_8_2)

            res5_4 = res5_3 + eltwise5_8

            conv5a = self._conv_layer(res5_4, name='conv5a')
            slice5a_1, slice5a_2 = tf.split(conv5a, 2, 3)
            eltwise5a = tf.maximum(slice5a_1, slice5a_2)

            conv5 = self._conv_layer(eltwise5a, name='conv5')
            slice5_1, slice5_2 = tf.split(conv5, 2, 3)
            eltwise5 = tf.maximum(slice5_1, slice5_2)

            eltwise55 = tf.concat([eltwise5,eltwise5a],3)

            pool4 = tf.nn.max_pool(eltwise5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME')
            pool4_1 = tf.nn.max_pool(eltwise4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME')
            pool4_2 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME')
            pool44 = tf.concat([pool4,pool4_1,pool4_2],3)

            pool4_transposed = tf.transpose(pool4, perm=[0, 3, 1, 2])
            # pool4_reshaped = tf.reshape(pool4_transposed, shape=[tf.shape(pool4)[0],-1])
            fc1 = self._fc_layer(pool4_transposed, name="fc1")
            slice_fc1_1, slice_fc1_2 = tf.split(fc1, 2, 1)
            eltwise_fc1 = tf.maximum(slice_fc1_1, slice_fc1_2)

            eltwise_fc11 = eltwise_fc1

            if P == True:
                eltwise_fc11 = tf.reshape(eltwise_fc11,[eltwise_fc11.shape[0],1,1,eltwise_fc11.shape[1]])
                return eltwise11, eltwise22, eltwise33, eltwise55, pool44, eltwise_fc11, pool4
            else:
                return eltwise11, eltwise22, eltwise33, eltwise55, pool4, fc1
        #DEEPFACE NET ENDS---

        #DEEPFACE OPS BEGINS---
    def  _conv_layer(self, input_, output_dim=128,
                    k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02,
                    name="conv2d"):
        '''
        进行一次卷积层运算
        Note: currently kernel size and input output channel num are decided by loaded filter weights.
        #only strides are decided by calling param.'''
        with tf.variable_scope(name) as scope:
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(input_, filt, strides=[1, d_h, d_w, 1], padding='SAME')
            conv_biases = self.get_bias(name)
            return tf.nn.bias_add(conv, conv_biases)
            return conv

    def _fc_layer(self, bottom, name="fc1", num_classes=None):
        with tf.variable_scope(name) as scope:
            #shape = bottom.get_shape().as_list()
            if name == 'fc1':
                filt = self.get_fc_weight(name)
                bias = self.get_bias(name)
            reshaped_bottom = tf.reshape(bottom,[tf.shape(bottom)[0],-1])
            return tf.matmul(reshaped_bottom, filt) + bias

    def get_conv_filter(self, name):

        init = tf.constant_initializer(value=self.data_dict[name][0],
                                       dtype=tf.float32)
        shape = self.data_dict[name][0].shape     #data_dict[name][0]
        var = tf.get_variable(name="filter", initializer=init, shape=shape)
        return var

    def get_bias(self, name, num_classes=None):
        bias_wights = self.data_dict[name][1]    #data_dict[name][1]
        shape = self.data_dict[name][1].shape
        init = tf.constant_initializer(value=bias_wights,
                                       dtype=tf.float32)
        var = tf.get_variable(name="biases", initializer=init, shape=shape)
        return var

    def get_fc_weight(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][0],
                                       dtype=tf.float32)
        shape = self.data_dict[name][0].shape      #data_dict[name][0]
        var = tf.get_variable(name="weights", initializer=init, shape=shape)
        return var

    #DEEPFACE OPS ENDS---

    def save(self, checkpoint_dir, step):
        if self.lstm:
            P = 'lstm'
        else:
            P = 'concat'
        model_name = "DCGAN.model"
        model_dir = "%s_%s_%s_%s" % (self.dataset_name, self.batch_size, self.time_size, P)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print(" [*] Saving checkpoints...at step " + str(step))
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        if self.lstm:
            P = 'lstm'
        else:
            P = 'concat'
        model_dir = "%s_%s_%s_%s" % (self.dataset_name, self.batch_size, self.time_size,P)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            #self.saver.restore(self.sess,ckpt.model_checkpoint_path)
            print(" [*] Success to read {}".format(ckpt_name),P)
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            return False

def main(_):

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    config = tf.ConfigProto(allow_soft_placement=True)
    # allow_soft_placement=True能让tensorflow遇到无法用GPU跑的数据时，自动切换成CPU进行。
    # log_device_placement则记录一些日志。
    config.gpu_options.allocator_type = 'BFC'  #使用BFC算法
    # config.gpu_options.per_process_gpu_memory_fraction = 0.90  #程序最多只能占用指定gpu90%的显存
    config.gpu_options.allow_growth = True   #程序按需申请内存

    with tf.Session(config=config) as sess:
        dcgan = DCGAN(sess,
                      batch_size=FLAGS.batch_size,
                      output_size=FLAGS.output_size,
                      dataset_name=FLAGS.dataset,
                      checkpoint_dir=FLAGS.checkpoint_dir,
                      )
        start_time_build = time.time()  #时间计算
        dcgan.build_model()
        print("build model costs %s seconds" %(time.time() - start_time_build)) #计算程序的运行时间
        dcgan.train(FLAGS)

if __name__ == '__main__':
    tf.app.run()