#!/usr/bin/python
# -*- coding: UTF-8 -*-
# -- coding: UTF-8
#coding=utf-8
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Routine for decoding the CIFAR-10 binary file format."""
#!/usr/bin/python
# -*- coding: UTF-8 -*-
# -- coding: UTF-8
#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

#from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
from PIL import Image
import random
import re
import csv

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 128
# Global constants describing the CIFAR-10 data set.
EYE_H = 40; EYE_W = 40;
NOSE_H = 32; NOSE_W = 40;
MOUTH_H = 32; MOUTH_W = 48;
re_pose = re.compile('_\d{3}_')
re_poseIllum = re.compile('_\d{3}_\d{2}_')


class MultiPIE():
    """Reads and parses examples from MultiPIE data filelist
    """
    def __init__(self, datasplit='train', Random=True, LOAD_60_LABEL=False, MIRROR_TO_ONE_SIDE=True, RANDOM_VERIFY=False,
                 GENERATE_MASK=False, testing = True):

        # if not testing:
        self.dir = './data1/MultiPIE/totally'
        # else:
        self.dir_test = './data1/MultiPIE/totally'

        self.csvpath  = './data1/MultiPIE/train.csv'
        self.test_dir = './data1/MultiPIE/test.csv'

        self.split = datasplit
        self.random = Random
        self.seed = None
        self.LOAD_60_LABEL = LOAD_60_LABEL
        self.MIRROR_TO_ONE_SIDE = MIRROR_TO_ONE_SIDE
        self.RANDOM_VERIFY = RANDOM_VERIFY
        self.GENERATE_MASK = GENERATE_MASK
        self.cameraPositions = {'24_0': (+90, '10'),'01_0' : (+75, '08'), '20_0' : (+60, '08'), '19_0' : (+45, '09'), '04_1' : (+30, '07'), '05_0' : (+15, '06'), #left 右脸
                    '05_1' : (0,'06'), #center
                    '14_0' : (-15,'06'), '13_0' : (-30, '05'), '08_0' : (-45, '15'),'09_0' : (-60, '15'),'12_0' : (-75, '15'),'11_0' : (-90, '15')} #right 左脸

        # if not testing: #训练阶段
        self.indices = open(self.csvpath, 'r').read().splitlines() #打开train.csv
        self.size = len(self.indices) #40*4=160

        # else:
        self.indices_test = open(self.test_dir, 'r').read().splitlines() #打开test.csv
        self.test_size = len(self.indices_test) #20*4=80

        self.idx = 0

    def test_batch(self, test_batch_size,time_size,idxx,jdxx):  #test_batch_size=100
        test_batch_size = min(test_batch_size, len(self.indices_test)) #min(150 ,80)
        images = np.empty([test_batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]) #(80,128,128,3)
        labels = np.empty([test_batch_size, IMAGE_SIZE, IMAGE_SIZE, 3])
        # pose = np.empty([test_batch_size],dtype=np.int32)
        filenames = list()
        poses = list()

        if idxx==0 and jdxx==0:
            for i in range(len(self.indices_test)//12):
                a1 = self.indices_test[i*12 : (i+1)*12]
                random.shuffle(a1)

                for ii in range(8):
                    b = a1[ii].find('110')
                    c = a1[ii].find('240')
                    if b != -1:
                        t = a1[ii]
                        a1[ii] = a1[11]
                        a1[11] = t
                    if c != -1:
                        t = a1[ii]
                        a1[ii] = a1[10]
                        a1[10] = t

                self.indices_test[i*12 : (i+1)*12] = a1

        a = ['050','041','190','200','010','240']
        b = ['140','130','080','090','120','110']
        for i in range(test_batch_size):
            idx = idxx * (time_size * test_batch_size) + (jdxx * test_batch_size + i)
            # idx = idxx * (time_size * 6) + (jdxx * 6 + i)
            images[i, ...] = self.load_image_test(self.indices_test[idx])
            filename = self.indices_test[idx]

            # labels[i, ...], labelname = self.load_label_mask(filename)

            filenames.append(filename)
            pose = abs(self.findPose(filename))
            poses.append(pose)
        return images, filenames, None, None, poses, None

    def test_batch1(self, test_batch_size,time_size,idxx,jdxx):  #test_batch_size=100
        test_batch_size = min(test_batch_size, len(self.indices_test)) #min(150 ,80)
        images = np.empty([test_batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]) #(80,128,128,3)
        filenames = list()
        poses = list()

        a = [['050_06','041_09','190_09','200_08','010_08','240_10'],['140_06','130_05','080_15','090_15','120_15','110_15']]

        for i in range(test_batch_size):
            idx = idxx * (time_size * test_batch_size) + (jdxx * test_batch_size + i)
            filename = self.indices_test[idx][0:10] + a[i][jdxx] + self.indices_test[idx][16:]
            images[i, ...] = self.load_image_test(filename)

            filenames.append(filename)
            pose = abs(self.findPose(filename))
            poses.append(pose)
        return images, filenames, None, None, poses, None

    def next_image_and_label_mask_batch1(self, batch_size,time_size,idxx,jdxx):

        filenames = list()
        labelnames = list()
        poses = list()
        images = np.empty([batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]) #[10, 128, 128, 3]
        labels = np.empty([batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]) #[10, 128, 128, 3]
        idenlabels = np.empty([batch_size],dtype=np.int32)

        # a = [['050', '041', '190', '200', '010', '240'], ['140', '130', '080', '090', '120', '110']]
        a = [['050_06','041_09','190_09','200_08','010_08','240_10'],['140_06','130_05','080_15','090_15','120_15','110_15']]
        # b = ['140','130','080','090','120','110']
        # for j in range(jdxx):
        for i in range(batch_size):
            idx = idxx * (time_size * batch_size) + (jdxx * batch_size + i)
            filename = self.indices[idx][0:10] + a[i][jdxx] + self.indices[idx][16:]
            images[i, ...] = self.load_image(filename)

            filenames.append(filename)

            labels[i, ...], labelname = self.load_label_mask(filename)

            labelnames.append(labelname)
            pose = abs(self.findPose(filename))
            poses.append(pose)
            identity = int(filename[0:3])
            idenlabels[i] = identity
        return images, labels, poses, idenlabels,filenames,labelnames


    def next_image_and_label_mask_batch(self, batch_size,time_size,idxx,jdxx):
        """Construct a batch of images and labels masks.
        Args:
        batch_size: Number of images per batch.
        shuffle: boolean indicating whether to use a shuffling queue.
        Returns:
        ndarray feed.
        images: Images. 4D of [batch_size, height, width, 6] size.
        labels: Labels. 4D of [batch_size, height, width, 3] size.
        masks: masks. 4D of [batch_size, height, width, 3] size.
        verifyImages: Images. 4D of [batch_size, height, width, 3] size.
        verifyLabels: 1D of [batch_size] 0 / 1 classification label
        """
        assert batch_size >= 1
        filenames = list()
        labelnames = list()
        poses = list()
        images = np.empty([batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]) #[10, 128, 128, 3]
        labels = np.empty([batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]) #[10, 128, 128, 3]
        # pose = np.empty([batch_size],dtype=np.int32)
        idenlabels = np.empty([batch_size],dtype=np.int32)
        if idxx==0 and jdxx==0:
            for i in range(len(self.indices)//12):
                a1 = self.indices[i*12 : (i+1)*12]
                random.shuffle(a1)
                self.indices[i*12: (i+1)*12] = a1

        for i in range(batch_size):
            idx = idxx * (time_size * batch_size) + (jdxx * batch_size + i)
            images[i, ...] = self.load_image(self.indices[idx])

            filename = self.indices[idx]
            filenames.append(filename)

            labels[i, ...], labelname = self.load_label_mask(filename)

            # labels[i, ...],labelname = self.load_label_mask(filename)
            labelnames.append(labelname)
            pose = abs(self.findPose(filename))
            poses.append(pose)
            identity = int(filename[0:3])
            idenlabels[i] = identity
        return images, labels, poses, idenlabels,filenames,labelnames

    def updateidx(self):
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0
    def load_image(self, filename):
            """
            Load input image & codemap and preprocess:
            - cast to float
            - subtract mean divide stdadv
            - concatenate together
            """
            im = Image.open(self.dir + os.sep + filename)
            in_ = np.array(im, dtype=np.float32) # (128,128,3)
            in_ /= 256
            return in_

    def load_image_test(self, filename):
            im = Image.open(self.dir_test + os.sep + filename)
            in_ = np.array(im, dtype=np.float32) # (128,128,3)
            in_ /= 256
            return in_

    def load_label_mask(self, filename, labelnum=-1):
        _, labelname = self.findSameIllumCodeLabelpath(filename) # input:207_02_01_090_15_cropped.png output:207_02_01_051_15_code.png, 207_02_01_051_15_cropped.png
        im = Image.open(self.dir + os.sep + labelname)
        label = np.array(im, dtype=np.float32) #(128,128,3)
        label /= 256
        return label,labelname

    def findBestIllumCodeImagepath(self, fullpath):
        span = re_pose.search(fullpath).span()
        camPos = list(fullpath[span[0]+1:span[1]-1])
        camPos.insert(2,'_')
        camPos = ''.join(camPos)
        #get 01_0 like string
        bestIllum = self.cameraPositions[camPos][1]

        labelpath = list(fullpath)
        labelpath[span[1]:span[1]+2] = bestIllum[:]
        labelpath = ''.join(labelpath)
        codepath = str(labelpath).replace('cropped', 'code')
        return (codepath, labelpath)

    def findSameIllumCodeLabelpath(self, fullpath):
        span = re_poseIllum.search(fullpath).span() #re_poseIllum = re.compile('_\d{3}_\d{2}_')
        tempath = list(fullpath)
        tempath[span[0]+1:span[0]+7] = '051_06'
        labelpath = ''.join(tempath)
        codepath = str(labelpath).replace('cropped', 'code')
        return (codepath, labelpath)

    def findPose(self, fullpath):
        span = re_pose.search(fullpath).span()  #re_pose = re.compile('_\d{3}_') #\d 表示[0-9] span() 返回匹配的区间[9,14] 左闭右开
        camPos = list(fullpath[span[0]+1:span[1]-1]) #'201_01_01_010_08_cropped.png'
        camPos.insert(2,'_') #['0', '1', '_', '0']
        camPos = ''.join(camPos) #'01_0'
        #get 01_0 like string
        return self.cameraPositions[camPos][0] #+75
