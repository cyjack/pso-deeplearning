#!/usr/bin/env python
# coding: utf-8


import h5py
import numpy as np
import os,random
from keras.layers import Input,Reshape,ZeroPadding2D,Conv2D,Dropout,Flatten,Dense,Activation,MaxPooling2D,AlphaDropout
from keras import layers
import keras.models as Model
from keras.regularizers import *
from keras.optimizers import adam
import seaborn as sns
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import gc
# get_ipython().run_line_magic('matplotlib', 'inline')
os.environ["KERAS_BACKEND"] = "tensorflow"
from pyswarm import pso

# # **数据集处理**



############################################
#由于硬件限制，无法使用完整数据集，因此我从完整数据集中抽取出部分数据，并分割成24个部分
#每部分对应一种调制，有1200*26=31200条数据
#因此，目前数据集大小为748800*1024*2
############################################
for i in range(0,24): #24个数据集文件
    ########打开文件#######
    filename = './ExtractDataset_1w/part'+str(i) + '.h5'
    print(filename)
    f = h5py.File(filename,'r')
    ########读取数据#######
    X_data = f['X'][:]
    Y_data = f['Y'][:]
    Z_data = f['Z'][:]
    f.close()
    #########分割训练集和测试集#########
    #每读取到一个数据文件就直接分割为训练集和测试集，防止爆内存
    n_examples = X_data.shape[0]
    n_train = int(n_examples * 0.7)   #70%训练样本
    train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)#随机选取训练样本下标
    test_idx = list(set(range(0,n_examples))-set(train_idx))        #测试样本下标
    if i == 0:
        X_train = X_data[train_idx]
        Y_train = Y_data[train_idx]
        Z_train = Z_data[train_idx]
        X_test = X_data[test_idx]
        Y_test = Y_data[test_idx]
        Z_test = Z_data[test_idx]
    else:
        X_train = np.vstack((X_train, X_data[train_idx]))
        Y_train = np.vstack((Y_train, Y_data[train_idx]))
        Z_train = np.vstack((Z_train, Z_data[train_idx]))
        X_test = np.vstack((X_test, X_data[test_idx]))
        Y_test = np.vstack((Y_test, Y_data[test_idx]))
        Z_test = np.vstack((Z_test, Z_data[test_idx]))
print('训练集X维度：',X_train.shape)
print('训练集Y维度：',Y_train.shape)
print('训练集Z维度：',Z_train.shape)
print('测试集X维度：',X_test.shape)
print('测试集Y维度：',Y_test.shape)
print('测试集Z维度：',Z_test.shape)


##查看数据是否正常
sample_idx = 520 #随机下标
print('snr:',Z_train[sample_idx])
print('Y',Y_train[sample_idx])
plt_data = X_train[sample_idx].T
plt.figure(figsize=(15,5))
plt.plot(plt_data[0])
plt.plot(plt_data[1],color = 'red')
plt.show()

# # **建立模型**

"""建立模型"""
classes = ['32PSK',
 '16APSK',
 '32QAM',
 'FM',
 'GMSK',
 '32APSK',
 'OQPSK',
 '8ASK',
 'BPSK',
 '8PSK',
 'AM-SSB-SC',
 '4ASK',
 '16PSK',
 '64APSK',
 '128QAM',
 '128APSK',
 'AM-DSB-SC',
 'AM-SSB-WC',
 '64QAM',
 'QPSK',
 '256QAM',
 'AM-DSB-WC',
 'OOK', 
 '16QAM']
data_format = 'channels_first'

def residual_stack(Xm,kennel_size,Seq,pool_size):
    #1*1 Conv Linear
    Xm = Conv2D(32, (1, 1), padding='same', name=Seq+"_conv1", kernel_initializer='glorot_normal',data_format=data_format)(Xm)
    #Residual Unit 1
    Xm_shortcut = Xm
    Xm = Conv2D(32, kennel_size, padding='same',activation="relu",name=Seq+"_conv2", kernel_initializer='glorot_normal',data_format=data_format)(Xm)
    Xm = Conv2D(32, kennel_size, padding='same', name=Seq+"_conv3", kernel_initializer='glorot_normal',data_format=data_format)(Xm)
    Xm = layers.add([Xm,Xm_shortcut])
    Xm = Activation("relu")(Xm)
    #Residual Unit 2
    Xm_shortcut = Xm
    Xm = Conv2D(32, kennel_size, padding='same',activation="relu",name=Seq+"_conv4", kernel_initializer='glorot_normal',data_format=data_format)(Xm)
    X = Conv2D(32, kennel_size, padding='same', name=Seq+"_conv5", kernel_initializer='glorot_normal',data_format=data_format)(Xm)
    Xm = layers.add([Xm,Xm_shortcut])
    Xm = Activation("relu")(Xm)
    #MaxPooling
    Xm = MaxPooling2D(pool_size=pool_size, strides=pool_size, padding='valid', data_format=data_format)(Xm)
    return Xm

#fine_tuning,nnunits,dropout,learning_rate
lb=[0.2,0.0008]
ub=[0.5,0.003]

def model_design(x):
    print(f"Units : {x[0]}, Learning Rate : {x[1]}")
    in_shp = X_train.shape[1:]   #每个样本的维度[1024,2]
    #学习率和dropout调节
    # if x[1] < 0.003:
    #     learning_rate = 0.001
    # elif x[1] < 0.0075:
    #     learning_rate = 0.005
    # elif x[1] < 0.015:
    #     learning_rate = 0.01
    # elif x[1] < 0.035:
    #     learning_rate = 0.02
    # elif x[1] < 0.075:
    #     learning_rate = 0.05
    # elif x[1] < 0.125:
    #     learning_rate = 0.1
    # elif x[1] < 0.175:
    #     learning_rate = 0.15
    # else:
    #     learning_rate = 0.2

    #input layer
    Xm_input = Input(in_shp)
    Xm = Reshape([1,1024,2], input_shape=in_shp)(Xm_input)
    #Residual Srack
    Xm = residual_stack(Xm,kennel_size=(3,2),Seq="ReStk0",pool_size=(2,2))   #shape:(512,1,32)
    Xm = residual_stack(Xm,kennel_size=(3,1),Seq="ReStk1",pool_size=(2,1))   #shape:(256,1,32)
    Xm = residual_stack(Xm,kennel_size=(3,1),Seq="ReStk2",pool_size=(2,1))   #shape:(128,1,32)
    Xm = residual_stack(Xm,kennel_size=(3,1),Seq="ReStk3",pool_size=(2,1))   #shape:(64,1,32)
    Xm = residual_stack(Xm,kennel_size=(3,1),Seq="ReStk4",pool_size=(2,1))   #shape:(32,1,32)
    Xm = residual_stack(Xm,kennel_size=(3,1),Seq="ReStk5",pool_size=(2,1))   #shape:(16,1,32)

    #############################################################################
    #      多次尝试发现减少一层全连接层能使loss下降更快
    #      将AlphaDropout设置为0.3似乎比0.5效果更好
    #############################################################################
    #Full Con 1
    Xm = Flatten(data_format=data_format)(Xm)
    Xm = Dense(128, activation='selu', kernel_initializer='glorot_normal', name="dense1")(Xm)
    Xm = AlphaDropout(x[0])(Xm)#修改了dropout的值
    #Full Con 2
    Xm = Dense(len(classes), kernel_initializer='glorot_normal', name="dense2")(Xm)
    #SoftMax
    Xm = Activation('softmax')(Xm)
    #Create Model
    model = Model.Model(inputs=Xm_input,outputs=Xm)
    adam = keras.optimizers.Adam(lr=x[1], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    ##修改了学习率的值
    model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=["accuracy"])
    # model.summary()
    return model

# # **训练**

"""训练模型"""
#############################################################################
#      当val_loss连续10次迭代不再减小或总迭代次数大于100时停止
#      将最小验证损失的模型保存
#############################################################################

print(tf.test.gpu_device_name())

filepath = './model1/ResNet_Model_72w.h5'
def best_model(x,count=1):
    model = model_design(x)
    history = model.fit(X_train,
        Y_train,
        batch_size=1000,
        epochs=50,
        verbose=2,
        validation_data=(X_test, Y_test),
        #validation_split = 0.3,
        callbacks = [
            keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
        ])
    train_loss, train_acc = model.evaluate(X_train, Y_train, verbose=0)
    print(f"Train Accuracy:{train_acc} Train Loss: {train_loss}")

    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Test Accuracy:{test_acc} Test Loss: {test_loss}")

    model.save(f"./model/model-{count}-{round(test_acc, 3)}-{round(test_loss, 3)}--Units-{x[0]}--Learning_rate-{x[1]}")
    np.savetxt(f"data-{count}.csv", x, delimiter=',')

    count = count
    test_acc_list = []
    test_loss_list = []
    count_no = []
    test_units = []
    test_learning_rate = []
    if test_acc > 0.99 and count < 0:
        # Plot the graph
        count = count - 1
        count_no.append(count)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)
        test_units.append(x[0])
        test_learning_rate.append(x[1])
    global result
    result = pd.DataFrame()
    result["count_no"] = count_no
    result["Test_Acc"] = test_acc_list
    result["Test_Loss"] = test_loss_list
    result["Units"] = test_units
    result["Learning_rate"] = test_learning_rate
    #描述训练集的loss

    val_loss_list = history.history['val_loss']
    loss_list = history.history['loss']
    plt.plot(range(len(loss_list)), val_loss_list)
    plt.plot(range(len(loss_list)), loss_list)
    plt.show()

    return test_acc

xopt, fopt = pso(best_model, lb, ub)

print ("Loss:" + str(fopt))
# we re-load the best weights once training is finished
# model.load_weights(filepath)











# ##########从loss走势来看，预计loss还能继续下降，故再训练一次#######
# history = model.fit(X_train,
#     Y_train,
#     batch_size=1000,
#     epochs=100,
#     verbose=2,
#     validation_data=(X_test, Y_test),
#     #validation_split = 0.3,
#     callbacks = [
#         keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
#         keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
#     ])
#
# # we re-load the best weights once training is finished
# model.load_weights(filepath)


# # **测试**



def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# Plot confusion matrix
batch_size = 1024
test_Y_hat = model.predict(X_test, batch_size=1024)
conf = np.zeros([len(classes),len(classes)])
confnorm = np.zeros([len(classes),len(classes)])
for i in range(0,X_test.shape[0]):
    j = list(Y_test[i,:]).index(1)
    k = int(np.argmax(test_Y_hat[i,:]))
    conf[j,k] = conf[j,k] + 1
for i in range(0,len(classes)):
    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
plot_confusion_matrix(confnorm, labels=classes)





for i in range(len(confnorm)):
    print(classes[i],confnorm[i,i])





acc={}
Z_test = Z_test.reshape((len(Z_test)))
SNRs = np.unique(Z_test)
for snr in SNRs:
    X_test_snr = X_test[Z_test==snr]
    Y_test_snr = Y_test[Z_test==snr]
    
    pre_Y_test = model.predict(X_test_snr)
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,X_test_snr.shape[0]):    #该信噪比下测试数据量
        j = list(Y_test_snr[i,:]).index(1)   #正确类别下标
        j = classes.index(classes[j])
        k = int(np.argmax(pre_Y_test[i,:])) #预测类别下标
        k = classes.index(classes[k])
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
   
    plt.figure()
    plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))
    
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print ("Overall Accuracy %s: "%snr, cor / (cor+ncor))
    acc[snr] = 1.0*cor/(cor+ncor)





plt.plot(acc.keys(),acc.values())
plt.ylabel('ACC')
plt.xlabel('SNR')
plt.show()

