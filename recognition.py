# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import heapq
import face_recognition
import os,shutil

filepath = '/media/gpu/文档/lsf/TP-GAN/data1/syn/test_label_01/'  # 识别标签 即原图
filenames = sorted(os.listdir(filepath))

# k = list(range(0,135))
# for n in range(len(k)):
#     syn_sample_dir = './data1/222/222_3_17/' + str(k[n]) + '/'+ str(2)

# syn_sample_dir = './data1/test_best/' + str(74) + '_best/' + if name2[9:14] == '_051_':

val =['2','3','4','5','6']
for a in range(len(val)):
    syn_sample_dir = './data1/222/'
# syn_sample_dir = './data1//check_the_result/' + 'test1/' + str(6)
    filenames2 = sorted(os.listdir(syn_sample_dir + '/'))
    class_num = len(filenames2)
    count = 0
    # 依次将合成的图片与原图进行比较识别
    my_face_encodings = []
    for i in range(len(filenames)):  # 识别标签 即原图
        picture_of_me = face_recognition.load_image_file(filepath + filenames[i])
        my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]
        my_face_encodings.append([my_face_encoding])

    for name2 in filenames2:  # 待识别样本 即合成图像

        # if name2[9:14] == '_051_':
        #     del_file = syn_sample_dir + '/' + name2  # 当代码和要删除的文件不在同一个文件夹时，必须使用绝对路径
        #     os.remove(del_file)  # 删除文件

        diss = []
        unknown_picture = face_recognition.load_image_file(syn_sample_dir + '/' + name2)
        unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]
        for i in range(len(my_face_encodings)):
            results, dis = face_recognition.compare_faces(my_face_encodings[i], unknown_face_encoding)
            diss.append(dis[0])
            if results[0] == True:
                count = count + 1
                break

        # min_m = list(map(diss.index, heapq.nsmallest(1, diss)))
        # for j in range(len(min_m)):
        #     if filenames[min_m[j]][0:3] == name2[0:3]:  # [12:15]  [7:10]
        #         count = count + 1
        #         break
           # # else:
           # #     print('待识别为 %s,' % name2, '标签为 %s,' % filenames[min_m[j]])

    # print('count = %s,' % count, 'class_num = %s,' % class_num)
    True_rat = count / class_num
    print(val[a], ':count = %s,' % count, 'class_num = %s,' % class_num, 'True_rat = %s,' % True_rat, 'ok')

