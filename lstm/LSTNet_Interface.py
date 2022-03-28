#-*- coding:utf-8 -*-
from keras.layers import *
from keras.layers.core import  Lambda,Activation
from keras.models import K,Model,Input
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import numpy as np
from .guass import GaussianLayer

from keras_self_attention import SeqSelfAttention

from keras.callbacks import EarlyStopping
from keras.layers import merge,add
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from keras.utils import plot_model
from .keras_transformer.transformer import *
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.utils import to_categorical
from keras.callbacks import TensorBoard

from keras.regularizers import l1

from keras.models import model_from_json

from .layer_utils import AttentionLSTM

from .LSTMCNN import Highway



#进行配置，每个GPU使用60%上限现存
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)
# 设置session
# KTF.set_session(session)


#设定为自增长
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# session = tf.Session(config=config)
# KTF.set_session(session)

def create_dataset(dataset, look_back,skip):
    '''
    对数据进行处理
    '''
    dataX,dataX2,dataY = [],[],[]
    #len(dataset)-1 不必要 但是可以避免某些状况下的bug
    for i in range(look_back*skip,len(dataset)-1):
        dataX.append(dataset[(i-look_back):i,:])
        dataY.append(dataset[i, :])
        temp=[]
        for j in range(i-look_back*skip,i,skip):
            temp.append(dataset[j,:])
        dataX2.append(temp)

    TrainX = np.array(dataX)
    TrainX2 = np.array(dataX2)
    TrainY = np.array(dataY)
    return TrainX, TrainX2 , TrainY

SINGLE_ATTENTION_VECTOR = False
def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = inputs
    #a = Permute((2, 1))(inputs)
    #a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(input_dim, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((1, 2), name='attention_vec')(a)

    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul


def LSTNet(trainX1,trainX2,trainY,config):

    input1 = Input(shape=(trainX1.shape[1], trainX1.shape[2]))
    conv1 = Conv1D(filters=64, kernel_size=6, strides=1, activation='relu')  # for input1

    # It's a probelm that I can't find any way to use the same Conv1D layer to train the two inputs,
    # conv2 = Conv1D(filters=48, kernel_size=6 , strides=1, activation='relu')  # for input2
    # conv2.set_weights(conv1.get_weights())  # at least use same weight

    conv1out = conv1(input1)
    # layer_t = AvgPool1D(pool_size=3)(conv1out)
    # conv1out = Conv1D(filters=72, kernel_size=3, activation='relu')(conv1out)
    # layer_t = Flatten()(layer_t)
    # attention_mul = attention_3d_block(layer_t)

    # attention_mul = Flatten()(attention_mul)
    # conv1out = Dropout(0.4)(conv1out)
    lstm1out = LSTM(32,return_sequences=True)(conv1out)
    # lstm1out = Dropout(0.4)(lstm1out)
    # lstm1out = Attention(32)(lstm1out)
    attention_mul = attention_3d_block(lstm1out)
    attention_mul = Flatten()(attention_mul)

    # lstm1out = Dropout(config.dropout)(lstm1out)

    input2 = Input(shape=(trainX2.shape[1], trainX2.shape[2]))
    # conv2out = conv2(input2)
    # lstm2out = LSTM(64)(conv2out)
    # lstm2out = Dropout(config.dropout)(lstm2out)

    # attention_mul = attention_3d_block(lstm1out)
    # attention_mul = Flatten()(attention_mul)

    lstm_out = attention_mul
    output = Dense(1,activation='sigmoid')(lstm_out)

    #highway  使用Dense模拟AR自回归过程，为预测添加线性成份，同时使输出可以响应输入的尺度变化。
    highway_window = config.highway_window
    #截取近3个窗口的时间维 保留了所有的输入维度
    z = Lambda(lambda k: k[:, -highway_window:, :])(input1)
    z = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1)))(z)
    z = Lambda(lambda k: K.reshape(k, (-1, highway_window*trainX1.shape[2])))(z)
    # z = Flatten()(z)
    z = Dense(1)(z)
    # z = Lambda(lambda k: K.reshape(k, (-1, trainX1.shape[2])))(z)

    output = add([output,z])

    output = Activation('sigmoid')(output)
    output = Dense(1)(output)
    model = Model(inputs=[input1,input2], outputs=output)

    return  model

from keras.callbacks import LearningRateScheduler, ModelCheckpoint



def LSTNet2(trainX1, trainX2, trainY, config):
    input1 = Input(shape=(trainX1.shape[1], trainX1.shape[2]))
    conv1 = Conv1D(filters=64, kernel_size=6, strides=1, activation='relu')  # for input1
    # It's a probelm that I can't find any way to use the same Conv1D layer to train the two inputs,
    # conv2 = Conv1D(filters=48, kernel_size=6 , strides=1, activation='relu')  # for input2
    # conv2.set_weights(conv1.get_weights())  # at least use same weight

    conv1out = conv1(input1)
    # layer_t = AvgPool1D(pool_size=3)(conv1out)
    # layer_t = Flatten()(layer_t)
    # attention_mul = attention_3d_block(layer_t)

    # attention_mul = Flatten()(attention_mul)

    lstm1out = LSTM(128, return_sequences=True)(conv1out)
    lstm1out = Dropout(0.5)(lstm1out)
    lstm1out = LSTM(64, return_sequences=False, activation='relu')(lstm1out)
    lstm1out = Dense(32, activation='relu')(lstm1out)
    lstm1out = Dropout(0.2)(lstm1out)
    lstm1out = Dense(16)(lstm1out)
    output = Dense(1)(lstm1out)

    # lstm1out = Attention(32)(lstm1out)
    #     attention_mul = attention_3d_block(lstm1out)
    #     attention_mul = Flatten()(attention_mul)

    # lstm1out = Dropout(config.dropout)(lstm1out)

    input2 = Input(shape=(trainX2.shape[1], trainX2.shape[2]))
    # conv2out = conv2(input2)
    # lstm2out = LSTM(64)(conv2out)
    # lstm2out = Dropout(config.dropout)(lstm2out)

    # attention_mul = attention_3d_block(lstm1out)
    # attention_mul = Flatten()(attention_mul)

    #     lstm_out = lstm1out
    #     output = Dense(trainY.shape[1],activation='sigmoid')(lstm_out)

    # highway  使用Dense模拟AR自回归过程，为预测添加线性成份，同时使输出可以响应输入的尺度变化。
    highway_window = config.highway_window
    # 截取近3个窗口的时间维 保留了所有的输入维度
    z = Lambda(lambda k: k[:, -highway_window:, :])(input1)
    z = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1)))(z)
    z = Lambda(lambda k: K.reshape(k, (-1, highway_window * trainX1.shape[2])))(z)
    z = Dense(trainY.shape[1])(z)

    output = add([output, z])
    output = Activation('sigmoid')(output)
    output = Dense(1)(output)
    model = Model(inputs=[input1, input2], outputs=output)

    return model

def LSTNet3(trainX1, trainX2, trainY, config):
    input1 = Input(shape=(trainX1.shape[1], trainX1.shape[2]))
    conv1 = Conv1D(filters=64, kernel_size=6, strides=1, activation='relu')  # for input1

    conv1out = conv1(input1)

    lstm1out = LSTM(128, return_sequences=True)(conv1out)
    lstm1out = Dropout(0.5)(lstm1out)
    attention_mul = attention_3d_block(lstm1out)
    attention_mul = Flatten()(attention_mul)
    # lstm1out = LSTM(64, return_sequences=False, activation='relu')(attention_mul)
    # lstm1out = Dropout(0.5)(lstm1out)
    lstm1out = Dense(32, activation='relu')(attention_mul)
    lstm1out = Dropout(0.2)(lstm1out)
    lstm1out = Dense(16)(lstm1out)
    output = Dense(1)(lstm1out)

    # lstm1out = Attention(32)(lstm1out)
    #     attention_mul = attention_3d_block(lstm1out)
    #     attention_mul = Flatten()(attention_mul)

    # lstm1out = Dropout(config.dropout)(lstm1out)

    input2 = Input(shape=(trainX2.shape[1], trainX2.shape[2]))
    # conv2out = conv2(input2)
    # lstm2out = LSTM(64)(conv2out)
    # lstm2out = Dropout(config.dropout)(lstm2out)

    # attention_mul = attention_3d_block(lstm1out)
    # attention_mul = Flatten()(attention_mul)

    #     lstm_out = lstm1out
    #     output = Dense(trainY.shape[1],activation='sigmoid')(lstm_out)

    # highway  使用Dense模拟AR自回归过程，为预测添加线性成份，同时使输出可以响应输入的尺度变化。
    highway_window = config.highway_window
    # 截取近3个窗口的时间维 保留了所有的输入维度
    z = Lambda(lambda k: k[:, -highway_window:, :])(input1)
    z = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1)))(z)
    z = Lambda(lambda k: K.reshape(k, (-1, highway_window * trainX1.shape[2])))(z)
    z = Dense(trainY.shape[1])(z)

    output = add([output, z])
    output = Activation('sigmoid')(output)
    output = Dense(1)(output)
    model = Model(inputs=[input1, input2], outputs=output)

    return model


def LSTNet4(trainX1, trainX2, trainY, config):
    input1 = Input(shape=(trainX1.shape[1], trainX1.shape[2]))
    conv1 = Conv1D(filters=64, kernel_size=6, strides=1, activation='relu')  # for input1
    conv1out = conv1(input1)
    lstm1out = LSTM(128, return_sequences=True)(conv1out)
    lstm1out = Dropout(0.5)(lstm1out)
    lstm1out = LSTM(64, return_sequences=False, activation='relu')(lstm1out)
    lstm1out = Dense(16,activation='relu')(lstm1out)
    # output = Dense(1)(lstm1out)

    output = lstm1out
    input2 = Input(shape=(trainX2.shape[1], trainX2.shape[2]))


    # highway  使用Dense模拟AR自回归过程，为预测添加线性成份，同时使输出可以响应输入的尺度变化。
    highway_window = config.highway_window
    # 截取近3个窗口的时间维 保留了所有的输入维度
    z = Lambda(lambda k: k[:, -highway_window:, :])(input1)
    z = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1)))(z)
    z = Lambda(lambda k: K.reshape(k, (-1, highway_window * trainX1.shape[2])))(z)
    # z = Dense(trainY.shape[1])(z)

    output = concatenate([output, z])
    output = Dense(32)(output)
    output = Dropout(0.2)(output)
    output = Dense(16)(output)
    output = Dense(1)(output)
    model = Model(inputs=[input1, input2], outputs=output)

    return model

def LSTNet5(trainX1, trainX2, trainY, config):
    input1 = Input(shape=(trainX1.shape[1], trainX1.shape[2]))
    conv1 = Conv1D(filters=64, kernel_size=6, strides=1, activation='relu')  # for input1

    conv1out = conv1(input1)


    lstm1out = LSTM(128, return_sequences=True)(conv1out)
    lstm1out = Dropout(0.5)(lstm1out)
    lstm1out = LSTM(64, return_sequences=False, activation='relu')(lstm1out)
    lstm1out = Dense(32, activation='relu')(lstm1out)
    lstm1out = Dropout(0.2)(lstm1out)
    lstm1out = Dense(16)(lstm1out)
    output = Dense(1)(lstm1out)


    input2 = Input(shape=(trainX2.shape[1], trainX2.shape[2]))

    highway_window = config.highway_window
    # 截取近3个窗口的时间维 保留了所有的输入维度
    z = Lambda(lambda k: k[:, -highway_window:, :])(input1)
    z = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1)))(z)
    z = Lambda(lambda k: K.reshape(k, (-1, highway_window * trainX1.shape[2])))(z)
    z = Dense(trainY.shape[1])(z)

    output = add([output, z])
    # output = Activation('sigmoid')(output)
    output = Dense(1)(output)
    model = Model(inputs=[input1, input2], outputs=output)

    return model


class CalculateScoreMatrix(Layer):
    def __init__(self, output_dim=None, **kwargs):
        self.output_dim = output_dim
        super(CalculateScoreMatrix, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
        })
        return config

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                 shape=(input_shape[-1], self.output_dim),
                                 initializer='uniform',
                                 trainable=True)
        super(CalculateScoreMatrix, self).build(input_shape)

    def call(self, x):
        res = K.dot(x, self.kernel)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)  # 指定输出维度


def LSTNet6(trainX1, trainX2, trainY, config):

    input1 = Input(shape=(trainX1.shape[1], trainX1.shape[2]))
    conv1 = Conv1D(filters=64, kernel_size=6, strides=1, activation='relu')  # for input1



    conv1out = conv1(input1)


    lstm1out = LSTM(32,return_sequences=True)(conv1out)

    attention_mul = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=regularizers.l2(1e-4),
                       bias_regularizer=regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4,
                       name='Attention')(lstm1out)
    attention_mul = Flatten()(attention_mul)
    # lstm1out = Dropout(config.dropout)(lstm1out)

    input2 = Input(shape=(trainX2.shape[1], trainX2.shape[2]))
    # conv2out = conv2(input2)
    # lstm2out = LSTM(64)(conv2out)
    # lstm2out = Dropout(config.dropout)(lstm2out)

    # attention_mul = attention_3d_block(lstm1out)
    # attention_mul = Flatten()(attention_mul)

    lstm_out = attention_mul
    output = Dense(64)(lstm_out)
    output = Dropout(0.2)(output)
    output = Dense(32)(output)
    output = Dropout(0.2)(output)
    output = Dense(1)(output)

    #highway  使用Dense模拟AR自回归过程，为预测添加线性成份，同时使输出可以响应输入的尺度变化。
    highway_window = config.highway_window
    #截取近3个窗口的时间维 保留了所有的输入维度
    z = Lambda(lambda k: k[:, -highway_window:, :])(input1)
    z = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1)))(z)
    z = Lambda(lambda k: K.reshape(k, (-1, highway_window*trainX1.shape[2])))(z)
    # z = Flatten()(z)
    z = Dense(1)(z)
    # z = Lambda(lambda k: K.reshape(k, (-1, trainX1.shape[2])))(z)

    output = add([output,z])

    output = Activation('sigmoid')(output)
    output = Dense(1)(output)
    model = Model(inputs=[input1,input2], outputs=output)

    return  model

def LSTNet7(trainX1, trainX2, trainY, config):
    input1 = Input(shape=(trainX1.shape[1], trainX1.shape[2]))
    conv1 = Conv1D(filters=64, kernel_size=6, strides=1, activation='relu')  # for input1
    conv1out = conv1(input1)
    conv1out = add([conv1out,input1])
    lstm1out = LSTM(128, return_sequences=True)(conv1out)
    lstm1out = Dropout(0.5)(lstm1out)
    lstm1out1 = add([lstm1out,conv1out])
    lstm1out = LSTM(64, return_sequences=False, activation='relu')(lstm1out1)
    lstm1out = BatchNormalization()(lstm1out)
    lstm1out = add([lstm1out,lstm1out1])
    lstm1out = Dense(16,activation='relu')(lstm1out)
    # output = Dense(1)(lstm1out)

    output = lstm1out
    input2 = Input(shape=(trainX2.shape[1], trainX2.shape[2]))


    # highway  使用Dense模拟AR自回归过程，为预测添加线性成份，同时使输出可以响应输入的尺度变化。
    highway_window = config.highway_window
    # 截取近3个窗口的时间维 保留了所有的输入维度
    z = Lambda(lambda k: k[:, -highway_window:, :])(input1)
    z = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1)))(z)
    z = Lambda(lambda k: K.reshape(k, (-1, highway_window * trainX1.shape[2])))(z)
    # z = Dense(trainY.shape[1])(z)

    output = concatenate([output, z])
    output = Dense(32)(output)
    output = Dropout(0.2)(output)
    output = Dense(16)(output)
    output = Dense(1)(output)
    model = Model(inputs=[input1, input2], outputs=output)

    return model


def LSTNet8(trainX1, trainX2, trainY, config):
    input1 = Input(shape=(trainX1.shape[1], trainX1.shape[2]))
    input2 = Input(shape=(trainX2.shape[1], trainX2.shape[2]))

    conv1 = Conv1D(filters=64, kernel_size=3, strides=1, activation='sigmoid')  # for input1


    conv1out = conv1(input1)
    conv2out = Conv1D(filters=128, kernel_size=1, strides=1, activation='sigmoid')(conv1out)
    conv3out = Conv1D(filters=64, kernel_size=6, strides=1, activation='sigmoid')(input1)

    # lstm1out = LSTM(128, return_sequences=True)(conv2out)
    # lstm1out = Dropout(0.4)(lstm1out)
    lstm1out = LSTM(64, return_sequences=False,activation='sigmoid')(conv1out)
    # lstm1out = Dropout(0.4)(lstm1out)
    # lstm1out = concatenate([lstm1out,conv1out,conv3out],axis=1)


    # lstm1out = Flatten()(lstm1out)
    lstm1out = Dense(32,activation='relu')(lstm1out)
    # lstm1out = BatchNormalization()(lstm1out)
    # lstm1out = Activation('relu')(lstm1out)
    lstm1out = Dropout(0.2)(lstm1out)
    lstm1out = Dense(16)(lstm1out)
    output = Dense(1)(lstm1out)



    # X = Flatten()(X)
    # X = Dropout(0.5)(X)
    # output = Dense(1)(X)

    highway_window = config.highway_window
    # 截取近3个窗口的时间维 保留了所有的输入维度
    z = Lambda(lambda k: k[:, -highway_window:, :])(input1)
    z = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1)))(z)
    z = Lambda(lambda k: K.reshape(k, (-1, highway_window * trainX1.shape[2])))(z)
    # z = Flatten()(z)
    z = Dense(1)(z)
    # z = Lambda(lambda k: K.reshape(k, (-1, trainX1.shape[2])))(z)

    # output = add([output, z])


    output = Activation('sigmoid')(output)
    output = Dense(1)(output)
    model = Model(inputs=[input1, input2], outputs=output)

    # for classification, if n_out =1, add:
    # X = Activation('sigmoid')(X)

    # for classification, if n_out > 1, add:
    # X = Activation('softmax')(X)

    # Create model

    return model

def LSTNet9(trainX1, trainX2, trainY, config):
    input1 = Input(shape=(trainX1.shape[1], trainX1.shape[2]))
    input2 = Input(shape=(trainX2.shape[1], trainX2.shape[2]))

    conv1 = Conv1D(filters=64, kernel_size=3, strides=1, activation='sigmoid')  # for input1


    conv1out = conv1(input1)

    conv2out = Conv1D(filters=128, kernel_size=1, strides=1, activation='sigmoid')(conv1out)
    # conv3out = identity_block()
    # lstm1out = LSTM(128, return_sequences=True)(conv2out)
    # lstm1out = Dropout(0.4)(lstm1out)
    lstm1out = LSTM(64, return_sequences=True,activation='sigmoid')(conv2out)
    lstm1out = add([conv1out,lstm1out])
    lstm1out = LSTM(64)(lstm1out)
    # lstm1out = Dropout(0.4)(lstm1out)
    # lstm1out = concatenate([lstm1out,conv1out,conv3out],axis=1)

    # lstm1out = add([conv3out,conv1out])

    # lstm1out = Flatten()(lstm1out)
    lstm1out = Dense(32,activation='relu')(lstm1out)
    # lstm1out = BatchNormalization()(lstm1out)
    # lstm1out = Activation('relu')(lstm1out)
    lstm1out = Dropout(0.2)(lstm1out)
    lstm1out = Dense(16)(lstm1out)
    output = Dense(1)(lstm1out)



    # X = Flatten()(X)
    # X = Dropout(0.5)(X)
    # output = Dense(1)(X)

    highway_window = config.highway_window
    # 截取近3个窗口的时间维 保留了所有的输入维度
    z = Lambda(lambda k: k[:, -highway_window:, :])(input1)
    z = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1)))(z)
    z = Lambda(lambda k: K.reshape(k, (-1, highway_window * trainX1.shape[2])))(z)
    # z = Flatten()(z)
    z = Dense(1)(z)
    # z = Lambda(lambda k: K.reshape(k, (-1, trainX1.shape[2])))(z)

    # output = add([output, z])


    output = Activation('sigmoid')(output)
    output = Dense(1)(output)
    model = Model(inputs=[input1, input2], outputs=output)

    # for classification, if n_out =1, add:
    # X = Activation('sigmoid')(X)

    # for classification, if n_out > 1, add:
    # X = Activation('softmax')(X)

    # Create model

    return model

def LSTNet10(trainX1, trainX2, trainY, config):
    input1 = Input(shape=(trainX1.shape[1], trainX1.shape[2]))
    input2 = Input(shape=(trainX2.shape[1], trainX2.shape[2]))

    conv1 = Conv1D(filters=64, kernel_size=3, strides=1, activation='sigmoid')  # for input1


    conv1out = conv1(input1)

    conv2out = Conv1D(filters=128, kernel_size=1, strides=1, activation='sigmoid')(conv1out)
    # conv3out = identity_block()
    # lstm1out = LSTM(128, return_sequences=True)(conv2out)
    # lstm1out = Dropout(0.4)(lstm1out)
    lstm1out = LSTM(64, return_sequences=True,activation='sigmoid')(conv2out)
    lstm1out = add([conv1out,lstm1out])
    lstm1out = LSTM(64)(lstm1out)
    # lstm1out = Dropout(0.4)(lstm1out)
    # lstm1out = concatenate([lstm1out,conv1out,conv3out],axis=1)

    # lstm1out = add([conv3out,conv1out])

    # lstm1out = Flatten()(lstm1out)
    lstm1out = Dense(32,activation='relu')(lstm1out)
    # lstm1out = BatchNormalization()(lstm1out)
    # lstm1out = Activation('relu')(lstm1out)
    lstm1out = Dropout(0.2)(lstm1out)
    lstm1out = Dense(16)(lstm1out)
    output = Dense(1)(lstm1out)



    # X = Flatten()(X)
    # X = Dropout(0.5)(X)
    # output = Dense(1)(X)

    highway_window = config.highway_window
    # 截取近3个窗口的时间维 保留了所有的输入维度
    z = Lambda(lambda k: k[:, -highway_window:, :])(input1)
    z = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1)))(z)
    z = Lambda(lambda k: K.reshape(k, (-1, highway_window * trainX1.shape[2])))(z)
    # z = Flatten()(z)
    z = Dense(1)(z)
    # z = Lambda(lambda k: K.reshape(k, (-1, trainX1.shape[2])))(z)

    output = add([output, z])



    output = Activation('sigmoid')(output)
    output = Dense(1)(output)
    model = Model(inputs=[input1, input2], outputs=output)

    # for classification, if n_out =1, add:
    # X = Activation('sigmoid')(X)

    # for classification, if n_out > 1, add:
    # X = Activation('softmax')(X)

    # Create model

    return model


def LSTNet11(trainX1, trainX2, trainY, config):
    input1 = Input(shape=(trainX1.shape[1], trainX1.shape[2]))
    input2 = Input(shape=(trainX2.shape[1], trainX2.shape[2]))

    conv1 = Conv1D(filters=64, kernel_size=3, strides=1, activation='sigmoid')  # for input1

    conv1out = conv1(input1)

    # conv2out = Conv1D(filters=128, kernel_size=1, strides=1, activation='sigmoid')(conv1out)
    # conv3out = identity_block()
    # lstm1out = LSTM(128, return_sequences=True)(conv2out)
    # lstm1out = Dropout(0.4)(lstm1out)
    lstm1out = LSTM(128, return_sequences=True, activation='sigmoid')(conv1out)
    conv3out = Conv1D(filters=128, kernel_size=1, strides=1, activation='sigmoid')(conv1out)
    lstm1out = Add()([conv3out, lstm1out])
    lstm1out = LSTM(64)(lstm1out)
    # lstm1out = Dropout(0.4)(lstm1out)
    # lstm1out = concatenate([lstm1out,conv1out,conv3out],axis=1)

    # lstm1out = add([conv3out,conv1out])

    # lstm1out = Flatten()(lstm1out)
    lstm1out = Dense(32)(lstm1out)
    # lstm1out = BatchNormalization()(lstm1out)
    # lstm1out = Activation('relu')(lstm1out)
    lstm1out = Dropout(0.2)(lstm1out)
    lstm1out = Dense(16)(lstm1out)
    output = Dense(1)(lstm1out)

    # X = Flatten()(X)
    # X = Dropout(0.5)(X)
    # output = Dense(1)(X)

    highway_window = config.highway_window
    # 截取近3个窗口的时间维 保留了所有的输入维度
    z = Lambda(lambda k: k[:, -highway_window:, :])(input1)
    z = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1)))(z)
    z = Lambda(lambda k: K.reshape(k, (-1, highway_window * trainX1.shape[2])))(z)
    # z = Flatten()(z)
    z = Dense(1)(z)
    # z = Lambda(lambda k: K.reshape(k, (-1, trainX1.shape[2])))(z)

    output = add([output, z])

    output = Activation('sigmoid')(output)
    output = Dense(1)(output)
    model = Model(inputs=[input1, input2], outputs=output)

    # for classification, if n_out =1, add:
    # X = Activation('sigmoid')(X)

    # for classification, if n_out > 1, add:
    # X = Activation('softmax')(X)

    # Create model

    return model

def LSTNet12(trainX1, trainX2, trainY, config):
    input1 = Input(shape=(trainX1.shape[1], trainX1.shape[2]))
    input2 = Input(shape=(trainX2.shape[1], trainX2.shape[2]))

    conv1 = Conv1D(filters=128, kernel_size=6, strides=1, activation='sigmoid')  # for input1

    conv1out = conv1(input1)

    # conv2out = Conv1D(filters=128, kernel_size=1, strides=1, activation='sigmoid')(conv1out)
    # conv3out = identity_block()
    # lstm1out = LSTM(128, return_sequences=True)(conv2out)
    # lstm1out = Dropout(0.4)(lstm1out)
    lstm1out = LSTM(64, return_sequences=True, activation='sigmoid')(conv1out)
    conv3out = Conv1D(filters=64, kernel_size=1, strides=1, activation='sigmoid')(conv1out)
    lstm1out = Add()([conv3out, lstm1out])
    lstm1out = LSTM(128)(lstm1out)
    # lstm1out = Dropout(0.4)(lstm1out)
    # lstm1out = concatenate([lstm1out,conv1out,conv3out],axis=1)

    # lstm1out = add([conv3out,conv1out])

    # lstm1out = Flatten()(lstm1out)
    lstm1out = Dense(32)(lstm1out)
    # lstm1out = BatchNormalization()(lstm1out)
    # lstm1out = Activation('relu')(lstm1out)
    lstm1out = Dropout(0.2)(lstm1out)
    lstm1out = Dense(16)(lstm1out)
    output = Dense(1)(lstm1out)

    # X = Flatten()(X)
    # X = Dropout(0.5)(X)
    # output = Dense(1)(X)

    highway_window = config.highway_window
    # 截取近3个窗口的时间维 保留了所有的输入维度
    z = Lambda(lambda k: k[:, -highway_window:, :])(input1)
    z = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1)))(z)
    z = Lambda(lambda k: K.reshape(k, (-1, highway_window * trainX1.shape[2])))(z)
    # z = Flatten()(z)
    z = Dense(1)(z)
    # z = Lambda(lambda k: K.reshape(k, (-1, trainX1.shape[2])))(z)

    output = add([output, z])

    output = Activation('sigmoid')(output)
    output = Dense(1)(output)
    model = Model(inputs=[input1, input2], outputs=output)

    # for classification, if n_out =1, add:
    # X = Activation('sigmoid')(X)

    # for classification, if n_out > 1, add:
    # X = Activation('softmax')(X)

    # Create model

    return model

def LSTNet13(trainX1, trainX2, trainY, config):
    input1 = Input(shape=(trainX1.shape[1], trainX1.shape[2]))
    input2 = Input(shape=(trainX2.shape[1], trainX2.shape[2]))

    conv1 = Conv1D(filters=128, kernel_size=6, strides=1, activation=None)  # for input1

    conv1out = conv1(input1)
    conv1out = LeakyReLU()(conv1out)

    # conv2out = Conv1D(filters=128, kernel_size=1, strides=1, activation='sigmoid')(conv1out)
    # conv3out = identity_block()
    # lstm1out = LSTM(128, return_sequences=True)(conv2out)
    # lstm1out = Dropout(0.4)(lstm1out)
    lstm1out = LSTM(64, return_sequences=True, kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None), kernel_regularizer=regularizers.l2(0.001))(conv1out)
    # lstm1out = LeakyReLU()(lstm1out)
    # lstm1out = attention_3d_block(lstm1out)
    # attention_mul = Flatten()(attention_mul)
    conv3out = Conv1D(filters=64, kernel_size=1, strides=1,activation=None)(conv1out)
    conv3out = LeakyReLU()(conv3out)
    lstm1out = Add()([conv3out, lstm1out])
    lstm1out = LSTM(128, activation=None, kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None))(lstm1out)
    lstm1out = LeakyReLU()(lstm1out)
    lstm1out = Dropout(0.4)(lstm1out)
    # lstm1out = concatenate([lstm1out,conv1out,conv3out],axis=1)

    # lstm1out = add([conv3out,conv1out])

    # lstm1out = Flatten()(lstm1out)
    lstm1out = Dense(32)(lstm1out)
    # lstm1out = BatchNormalization()(lstm1out)
    # lstm1out = Activation('relu')(lstm1out)
    lstm1out = Dropout(0.2)(lstm1out)
    lstm1out = Dense(16)(lstm1out)
    output = Dense(1)(lstm1out)

    # X = Flatten()(X)
    # X = Dropout(0.5)(X)
    # output = Dense(1)(X)

    highway_window = config.highway_window
    # 截取近3个窗口的时间维 保留了所有的输入维度
    z = Lambda(lambda k: k[:, -highway_window:, :])(input1)
    z = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1)))(z)
    z = Lambda(lambda k: K.reshape(k, (-1, highway_window * trainX1.shape[2])))(z)
    # z = Flatten()(z)
    z = Dense(1)(z)
    # z = Lambda(lambda k: K.reshape(k, (-1, trainX1.shape[2])))(z)

    output = add([output, z])

    output = Activation('sigmoid')(output)
    output = Dense(1)(output)
    model = Model(inputs=[input1, input2], outputs=output)

    # for classification, if n_out =1, add:
    # X = Activation('sigmoid')(X)

    # for classification, if n_out > 1, add:
    # X = Activation('softmax')(X)

    # Create model

    return model

def LSTNet14(trainX1, trainX2, trainY, config):
    input1 = Input(shape=(trainX1.shape[1], trainX1.shape[2]))
    input2 = Input(shape=(trainX2.shape[1], trainX2.shape[2]))

    conv1 = Conv1D(filters=64, kernel_size=2, strides=1, activation='tanh')  # for input1

    conv1out = conv1(input1)
    # conv1out = LeakyReLU()(conv1out)

    conv3out = Conv1D(filters=128, kernel_size=4, strides=1,activation='tanh')(conv1out)
    # conv3out = LeakyReLU()(conv3out)
    conv3out = Flatten()(conv3out)

    # lstm1out = add([conv3out,conv1out])

    # lstm1out = Flatten()(lstm1out)
    lstm1out = Dense(32)(conv3out)
    # lstm1out = BatchNormalization()(lstm1out)
    # lstm1out = Activation('relu')(lstm1out)
    lstm1out = Dropout(0.2)(lstm1out)
    lstm1out = Dense(16)(lstm1out)
    output = Dense(1,activation='sigmoid')(lstm1out)

    # X = Flatten()(X)
    # X = Dropout(0.5)(X)
    # output = Dense(1)(X)

    highway_window = config.highway_window
    # 截取近3个窗口的时间维 保留了所有的输入维度
    z = Lambda(lambda k: k[:, -highway_window:, :])(input1)
    z = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1)))(z)
    z = Lambda(lambda k: K.reshape(k, (-1, highway_window * trainX1.shape[2])))(z)
    # z = Flatten()(z)
    z = Dense(1)(z)
    # z = Lambda(lambda k: K.reshape(k, (-1, trainX1.shape[2])))(z)

    # output = add([output, z])

    # output = Activation('sigmoid')(output)
    # output = Dense(1)(output)
    model = Model(inputs=[input1, input2], outputs=output)

    # for classification, if n_out =1, add:
    # X = Activation('sigmoid')(X)

    # for classification, if n_out > 1, add:
    # X = Activation('softmax')(X)

    # Create model

    return model

def LSTNet15(trainX1, trainX2, trainY, config):
    input1 = Input(shape=(trainX1.shape[1], trainX1.shape[2]))
    input2 = Input(shape=(trainX2.shape[1], trainX2.shape[2]))

    conv1 = Conv1D(filters=32, kernel_size=6, strides=1, activation=None)  # for input1

    conv1out = conv1(input1)
    conv1out = ELU()(conv1out)

    # conv2out = Conv1D(filters=128, kernel_size=1, strides=1, activation='sigmoid')(conv1out)
    # conv3out = identity_block()
    # lstm1out = LSTM(128, return_sequences=True)(conv2out)
    # lstm1out = Dropout(0.4)(lstm1out)
    lstm1out = LSTM(32, return_sequences=True,activation=None)(conv1out)
    lstm1out = ELU()(lstm1out)
    lstm1out = Dropout(0.3)(lstm1out)
    lstm1out = attention_3d_block(lstm1out)


    conv3out = Conv1D(filters=32, kernel_size=1, strides=1)(conv1out)
    lstm1out = Add()([conv3out, lstm1out])
    lstm1out = LSTM(32)(lstm1out)
    lstm1out = ELU()(lstm1out)

    # lstm1out = Dropout(0.4)(lstm1out)
    # lstm1out = concatenate([lstm1out,conv1out,conv3out],axis=1)

    # lstm1out = add([conv3out,conv1out])

    # lstm1out = Flatten()(lstm1out)    # lstm1out = Dense(32)(lstm1out)
    #     # # lstm1out = BatchNormalization()(lstm1out)
    #     # # lstm1out = Activation('relu')(lstm1out)
    #     # lstm1out = Dropout(0.2)(lstm1out)

    lstm1out = Dense(16)(lstm1out)
    output = Dense(1)(lstm1out)

    # X = Flatten()(X)
    # X = Dropout(0.5)(X)
    # output = Dense(1)(X)

    highway_window = config.highway_window
    # 截取近3个窗口的时间维 保留了所有的输入维度
    z = Lambda(lambda k: k[:, -highway_window:, :])(input1)
    z = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1)))(z)
    z = Lambda(lambda k: K.reshape(k, (-1, highway_window * trainX1.shape[2])))(z)
    # z = Flatten()(z)
    z = Dense(1)(z)
    # z = Lambda(lambda k: K.reshape(k, (-1, trainX1.shape[2])))(z)

    # output = add([output, z])

    output = Activation('relu')(output)
    output = Dense(1)(output)
    model = Model(inputs=[input1, input2], outputs=output)

    # for classification, if n_out =1, add:
    # X = Activation('sigmoid')(X)

    # for classification, if n_out > 1, add:
    # X = Activation('softmax')(X)

    # Create model

    return model

def LSTNet16(trainX1, trainX2, trainY, config):
    input1 = Input(shape=(trainX1.shape[1], trainX1.shape[2]))
    input2 = Input(shape=(trainX2.shape[1], trainX2.shape[2]))

    conv1 = Conv1D(filters=128, kernel_size=6, strides=1,activation='relu')  # for input1

    conv1out = conv1(input1)


    # conv2out = Conv1D(filters=128, kernel_size=1, strides=1, activation='sigmoid')(conv1out)
    # conv3out = identity_block()
    # lstm1out = LSTM(128, return_sequences=True)(conv2out)
    # lstm1out = Dropout(0.4)(lstm1out)
    lstm1out = LSTM(64, return_sequences=True,activation=None)(conv1out)
    lstm1out = ELU()(lstm1out)

    conv3out = Conv1D(filters=64, kernel_size=1, strides=1)(conv1out)
    conv3out = LeakyReLU()(conv3out)
    conv4out = Conv1D(filters=64, kernel_size=3, strides=1)(conv1out)
    conv4out = LeakyReLU()(conv4out)
    conv5out = Conv1D(filters=64, kernel_size=4, strides=1)(conv1out)
    conv5out = LeakyReLU()(conv5out)



    lstm1out = Concatenate(axis=1)([conv3out, lstm1out,conv4out,conv5out])
    lstm1out = LSTM(32,activation=None)(lstm1out)
    lstm1out = ELU()(lstm1out)


    # lstm1out = Dropout(0.4)(lstm1out)
    # lstm1out = concatenate([lstm1out,conv1out,conv3out],axis=1)

    # lstm1out = add([conv3out,conv1out])

    # lstm1out = Flatten()(lstm1out)
    # lstm1out = Dense(32)(lstm1out)
    # # lstm1out = BatchNormalization()(lstm1out)
    # # lstm1out = Activation('relu')(lstm1out)
    # lstm1out = Dropout(0.2)(lstm1out)
    lstm1out = Dense(16)(lstm1out)
    output = Dense(1)(lstm1out)

    # X = Flatten()(X)
    # X = Dropout(0.5)(X)
    # output = Dense(1)(X)

    highway_window = config.highway_window
    # 截取近3个窗口的时间维 保留了所有的输入维度
    z = Lambda(lambda k: k[:, -highway_window:, :])(input1)
    z = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1)))(z)
    z = Lambda(lambda k: K.reshape(k, (-1, highway_window * trainX1.shape[2])))(z)
    # z = Dense(16)(z)
    # z = Flatten()(z)
    z = Dense(1)(z)
    # z = Lambda(lambda k: K.reshape(k, (-1, trainX1.shape[2])))(z)

    output = add([output, z])

    # output = Activation('relu')(output)
    output = Dense(1)(output)
    model = Model(inputs=[input1, input2], outputs=output)

    # for classification, if n_out =1, add:
    # X = Activation('sigmoid')(X)

    # for classification, if n_out > 1, add:
    # X = Activation('softmax')(X)

    # Create model

    return model


def LSTNet17(trainX1, trainX2, trainY, config):
    input1 = Input(shape=(trainX1.shape[1], trainX1.shape[2]))
    input2 = Input(shape=(trainX2.shape[1], trainX2.shape[2]))

    conv1 = Conv1D(filters=128, kernel_size=6, strides=1,activation='relu')  # for input1

    conv1out = conv1(input1)


    # conv2out = Conv1D(filters=128, kernel_size=1, strides=1, activation='sigmoid')(conv1out)
    # conv3out = identity_block()
    # lstm1out = LSTM(128, return_sequences=True)(conv2out)
    # lstm1out = Dropout(0.4)(lstm1out)
    lstm1out = LSTM(64, return_sequences=True,activation=None)(conv1out)
    lstm1out = ELU()(lstm1out)

    conv3out = Conv1D(filters=64, kernel_size=1, strides=1)(conv1out)
    conv3out = LeakyReLU()(conv3out)
    conv4out = Conv1D(filters=64, kernel_size=3, strides=1)(conv1out)
    conv4out = LeakyReLU()(conv4out)
    conv5out = Conv1D(filters=64, kernel_size=4, strides=1)(conv1out)
    conv5out = LeakyReLU()(conv5out)



    lstm1out = Concatenate(axis=1)([conv3out, lstm1out,conv4out,conv5out])
    lstm1out = LSTM(32,activation=None)(lstm1out)
    lstm1out = ELU()(lstm1out)


    # lstm1out = Dropout(0.4)(lstm1out)
    # lstm1out = concatenate([lstm1out,conv1out,conv3out],axis=1)

    # lstm1out = add([conv3out,conv1out])

    # lstm1out = Flatten()(lstm1out)
    # lstm1out = Dense(32)(lstm1out)
    # # lstm1out = BatchNormalization()(lstm1out)
    # # lstm1out = Activation('relu')(lstm1out)
    # lstm1out = Dropout(0.2)(lstm1out)
    lstm1out = Dense(16)(lstm1out)
    output = Dense(1)(lstm1out)

    # X = Flatten()(X)
    # X = Dropout(0.5)(X)
    # output = Dense(1)(X)

    highway_window = config.highway_window
    # 截取近3个窗口的时间维 保留了所有的输入维度
    z = Lambda(lambda k: k[:, -highway_window:, :])(input1)
    z = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1)))(z)
    z = Lambda(lambda k: K.reshape(k, (-1, highway_window * trainX1.shape[2])))(z)
    # z = Dense(16)(z)
    # z = Flatten()(z)
    z = Dense(1)(z)
    # z = Lambda(lambda k: K.reshape(k, (-1, trainX1.shape[2])))(z)

    # output = add([output, z])
    #
    # # output = Activation('relu')(output)
    # output = Dense(1)(output)
    model = Model(inputs=[input1, input2], outputs=output)

    # for classification, if n_out =1, add:
    # X = Activation('sigmoid')(X)

    # for classification, if n_out > 1, add:
    # X = Activation('softmax')(X)

    # Create model

    return model



def LSTNet18(trainX1, trainX2, trainY, config):

    input1 = Input(shape=(trainX1.shape[1], trainX1.shape[2]))
    conv1 = Conv1D(filters=64, kernel_size=6, strides=1, activation='relu')  # for input1

    # It's a probelm that I can't find any way to use the same Conv1D layer to train the two inputs,
    # conv2 = Conv1D(filters=48, kernel_size=6 , strides=1, activation='relu')  # for input2
    # conv2.set_weights(conv1.get_weights())  # at least use same weight

    conv1out = conv1(input1)
    # layer_t = AvgPool1D(pool_size=3)(conv1out)
    # conv1out = Conv1D(filters=72, kernel_size=3, activation='relu')(conv1out)
    # layer_t = Flatten()(layer_t)
    # attention_mul = attention_3d_block(layer_t)

    # attention_mul = Flatten()(attention_mul)
    # conv1out = Dropout(0.4)(conv1out)
    lstm1out = LSTM(32,return_sequences=True)(conv1out)
    # lstm1out = Dropout(0.4)(lstm1out)
    # lstm1out = Attention(32)(lstm1out)
    lstm1out = SeqSelfAttention(attention_activation='sigmoid')(lstm1out)
    # lstm1out = Dropout(config.dropout)(lstm1out)

    input2 = Input(shape=(trainX2.shape[1], trainX2.shape[2]))
    # conv2out = conv2(input2)
    # lstm2out = LSTM(64)(conv2out)
    # lstm2out = Dropout(config.dropout)(lstm2out)

    # attention_mul = attention_3d_block(lstm1out)
    # attention_mul = Flatten()(attention_mul)

    lstm_out = lstm1out
    lstm_out = Flatten()(lstm_out)
    lstm_out = Dense(32,activation='relu')(lstm_out)
    lstm_out = Dropout(0.2)(lstm_out)
    lstm_out = Dense(16)(lstm_out)
    output = Dense(1)(lstm_out)

    #highway  使用Dense模拟AR自回归过程，为预测添加线性成份，同时使输出可以响应输入的尺度变化。
    highway_window = config.highway_window
    #截取近3个窗口的时间维 保留了所有的输入维度
    z = Lambda(lambda k: k[:, -highway_window:, :])(input1)
    z = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1)))(z)
    z = Lambda(lambda k: K.reshape(k, (-1, highway_window*trainX1.shape[2])))(z)
    # z = Flatten()(z)
    z = Dense(1)(z)
    # z = Lambda(lambda k: K.reshape(k, (-1, trainX1.shape[2])))(z)

    # output = add([output,z])
    #
    # output = Activation('sigmoid')(output)
    # output = Dense(1)(output)
    model = Model(inputs=[input1,input2], outputs=output)

    return  model

def LSTNet19(trainX1, trainX2, trainY, config):
    input1 = Input(shape=(trainX1.shape[1], trainX1.shape[2]))
    input2 = Input(shape=(trainX2.shape[1], trainX2.shape[2]))

    conv1 = Conv1D(filters=128, kernel_size=6, strides=1,activation='relu')  # for input1

    conv1out = conv1(input1)


    # conv2out = Conv1D(filters=128, kernel_size=1, strides=1, activation='sigmoid')(conv1out)
    # conv3out = identity_block()
    # lstm1out = LSTM(128, return_sequences=True)(conv2out)
    # lstm1out = Dropout(0.4)(lstm1out)
    conv3out = Conv1D(filters=64, kernel_size=1, strides=1)(conv1out)
    conv3out = LeakyReLU()(conv3out)
    conv4out = Conv1D(filters=64, kernel_size=3, strides=1)(conv1out)
    conv4out = LeakyReLU()(conv4out)
    conv5out = Conv1D(filters=64, kernel_size=4, strides=1)(conv1out)
    conv5out = LeakyReLU()(conv5out)



    lstm1out = Concatenate(axis=1)([conv3out,conv4out,conv5out])
    lstm1out = LSTM(32,activation=None,return_sequences=True)(lstm1out)
    lstm1out = ELU()(lstm1out)
    lstm1out = SeqSelfAttention(attention_activation='relu',attention_type='multiplicative')(lstm1out)
    lstm1out = Flatten()(lstm1out)

    # lstm1out = Dropout(0.4)(lstm1out)
    # lstm1out = concatenate([lstm1out,conv1out,conv3out],axis=1)

    # lstm1out = add([conv3out,conv1out])

    # lstm1out = Flatten()(lstm1out)
    # lstm1out = Dense(32)(lstm1out)
    # # lstm1out = BatchNormalization()(lstm1out)
    # # lstm1out = Activation('relu')(lstm1out)
    # lstm1out = Dropout(0.2)(lstm1out)
    lstm1out = Dense(16)(lstm1out)
    output = Dense(1)(lstm1out)

    # X = Flatten()(X)
    # X = Dropout(0.5)(X)
    # output = Dense(1)(X)

    highway_window = config.highway_window
    # 截取近3个窗口的时间维 保留了所有的输入维度
    z = Lambda(lambda k: k[:, -highway_window:, :])(input1)
    z = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1)))(z)
    z = Lambda(lambda k: K.reshape(k, (-1, highway_window * trainX1.shape[2])))(z)
    # z = Dense(16)(z)
    # z = Flatten()(z)
    z = Dense(1)(z)
    # z = Lambda(lambda k: K.reshape(k, (-1, trainX1.shape[2])))(z)

    output = add([output, z])

    # output = Activation('relu')(output)
    output = Dense(1)(output)
    model = Model(inputs=[input1, input2], outputs=output)

    # for classification, if n_out =1, add:
    # X = Activation('sigmoid')(X)

    # for classification, if n_out > 1, add:
    # X = Activation('softmax')(X)

    # Create model

    return model

def LSTNet20(trainX1, trainX2, trainY, config):
    input1 = Input(shape=(trainX1.shape[1], trainX1.shape[2]))
    input2 = Input(shape=(trainX2.shape[1], trainX2.shape[2]))

    conv1 = Conv1D(filters=128, kernel_size=6, strides=1, activation='relu')  # for input1

    conv1out = conv1(input1)
    #
    # # conv2out = Conv1D(filters=128, kernel_size=1, strides=1, activation='sigmoid')(conv1out)
    # # conv3out = identity_block()
    # # lstm1out = LSTM(128, return_sequences=True)(conv2out)
    # # lstm1out = Dropout(0.4)(lstm1out)
    # conv3out = Conv1D(filters=64, kernel_size=1, strides=1)(conv1out)
    # conv3out = LeakyReLU()(conv3out)
    # conv4out = Conv1D(filters=64, kernel_size=3, strides=1)(conv1out)
    # conv4out = LeakyReLU()(conv4out)
    # conv5out = Conv1D(filters=64, kernel_size=4, strides=1)(conv1out)
    # conv5out = LeakyReLU()(conv5out)
    # encoder_inputs = Input(shape=(trainX1.shape[1], trainX1.shape[2]))
    # conv1out = Conv1D(filters=128, kernel_size=6, strides=1, activation='relu')(encoder_inputs)
    #
    # conv3out = Conv1D(filters=64, kernel_size=1, strides=1)(conv1out)
    # conv3out = LeakyReLU()(conv3out)
    # conv4out = Conv1D(filters=64, kernel_size=3, strides=1)(conv1out)
    # conv4out = LeakyReLU()(conv4out)
    # conv5out = Conv1D(filters=64, kernel_size=4, strides=1)(conv1out)
    # conv5out = LeakyReLU()(conv5out)
    # lstm1out = Concatenate(axis=1)([conv3out,conv4out,conv5out])

    encoder = LSTM(64,
                   batch_input_shape=(1, trainX1.shape[1], trainX1.shape[2]),
                   stateful=False,
                   return_sequences=True,
                   return_state=True,
                   recurrent_initializer='glorot_normal')

    encoder_outputs, state_h, state_c = encoder(conv1out)
    encoder_states = [state_h, state_c]  # 'encoder_outputs' are ignored and only states are kept.

    # Decoder training, using 'encoder_states' as initial state.
    decoder_inputs = Input(shape=(trainX1.shape[1], trainX1.shape[2]))

    decoder_lstm_1 = LSTM(64,
                          batch_input_shape=(1,trainX1.shape[1], trainX1.shape[2]),
                          stateful=False,
                          return_sequences=True,
                          return_state=False,
                          dropout=0.2,
                          recurrent_dropout=0.2) # True

    decoder_lstm_2 = LSTM(32,  # to avoid "kernel run out of time" situation. I used 128.
                          stateful=False,
                          return_sequences=False,
                          return_state=False,
                          dropout=0.2,
                          recurrent_dropout=0.2,
                          recurrent_initializer='glorot_normal')

    decoder_outputs = decoder_lstm_2((decoder_lstm_1(encoder_outputs, initial_state=encoder_states)))
    # decoder_outputs = SeqSelfAttention(attention_activation='relu', attention_type='multiplicative')(decoder_outputs)

    decoder_outputs = Dropout(0.3)(decoder_outputs)
    decoder_dense = Dense(16)
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_outputs = Dense(1, activation='relu')(decoder_outputs)

    # highway_window = config.highway_window
    #
    # # 截取近3个窗口的时间维 保留了所有的输入维度
    # z = Lambda(lambda k: k[:, -highway_window:, :])(decoder_inputs)
    # z = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1)))(z)
    # z = Lambda(lambda k: K.reshape(k, (-1, highway_window * trainX1.shape[2])))(z)
    # # z = Dense(16)(z)
    # # z = Flatten()(z)
    # z = Dense(1)(z)
    # # z = Lambda(lambda k: K.reshape(k, (-1, trainX1.shape[2])))(z)
    #
    # output = add([decoder_outputs, z])
    #
    # # output = Activation('relu')(output)
    # output = Dense(1)(output)

    # training model
    training_model = Model([input1, decoder_inputs], decoder_outputs)
    training_model.compile(optimizer='adam', loss='mean_squared_error')

    return training_model


def LSTNet21(trainX1, trainX2, trainY, config):
    input1 = Input(shape=(trainX1.shape[1], trainX1.shape[2]))
    input2 = Input(shape=(trainX2.shape[1], trainX2.shape[2]))

    conv1 = Conv1D(filters=128, kernel_size=6, strides=1, activation='relu')  # for input1

    conv1out = conv1(input1)


    conv3out = Conv1D(filters=64, kernel_size=1, strides=1)(conv1out)
    conv3out = LeakyReLU()(conv3out)
    conv4out = Conv1D(filters=64, kernel_size=3, strides=1)(conv1out)
    conv4out = LeakyReLU()(conv4out)
    conv5out = Conv1D(filters=64, kernel_size=4, strides=1)(conv1out)
    conv5out = LeakyReLU()(conv5out)

    lstm1out = Concatenate(axis=1)([conv3out, conv4out, conv5out])
    lstm1out = Bidirectional(LSTM(32, activation=None, return_sequences=True))(lstm1out)
    lstm1out = ELU()(lstm1out)
    lstm1out = SeqSelfAttention(attention_activation='relu', attention_type='multiplicative')(lstm1out)
    lstm1out = Flatten()(lstm1out)


    lstm1out = Dense(16)(lstm1out)
    output = Dense(1)(lstm1out)



    highway_window = config.highway_window
    # 截取近3个窗口的时间维 保留了所有的输入维度
    z = Lambda(lambda k: k[:, -highway_window:, :])(input1)
    z = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1)))(z)
    z = Lambda(lambda k: K.reshape(k, (-1, highway_window * trainX1.shape[2])))(z)

    z = Dense(1)(z)

    output = add([output, z])


    output = Dense(1)(output)
    model = Model(inputs=[input1, input2], outputs=output)


    return model


def LSTNet22(trainX1, trainX2, trainY, config):
    input1 = Input(shape=(trainX1.shape[1], trainX1.shape[2]))
    input2 = Input(shape=(trainX2.shape[1], trainX2.shape[2]))

    conv1 = Conv1D(filters=128, kernel_size=6, strides=1, activation='relu')  # for input1

    conv1out = conv1(input1)

    conv3out = Conv1D(filters=64, kernel_size=1, strides=1)(conv1out)
    conv3out = LeakyReLU()(conv3out)
    conv4out = Conv1D(filters=64, kernel_size=3, strides=1)(conv1out)
    conv4out = LeakyReLU()(conv4out)
    conv5out = Conv1D(filters=64, kernel_size=4, strides=1)(conv1out)
    conv5out = LeakyReLU()(conv5out)

    lstm1out = Concatenate(axis=1)([conv3out, conv4out, conv5out])
    lstm1out = Bidirectional(LSTM(32, activation=None, return_sequences=True))(lstm1out)
    lstm1out = ELU()(lstm1out)
    lstm1out = SeqSelfAttention(attention_activation='relu', attention_type='multiplicative')(lstm1out)
    lstm1out = Flatten()(lstm1out)

    lstm1out = Dense(16)(lstm1out)
    output = Dense(1)(lstm1out)

    highway_window = config.highway_window
    # 截取近3个窗口的时间维 保留了所有的输入维度
    z = Lambda(lambda k: k[:, -highway_window:, 0])(input1)
    # z = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1)))(z)
    z = Lambda(lambda k: K.reshape(k, (-1, highway_window)))(z)

    z = Dense(1)(z)

    output = add([output, z])

    output = Dense(1)(output)
    model = Model(inputs=[input1, input2], outputs=output)

    return model


def LSTNet23(trainX1, trainX2, trainY, config):
    input1 = Input(shape=(trainX1.shape[1], trainX1.shape[2]))
    input2 = Input(shape=(trainX2.shape[1], trainX2.shape[2]))

    conv1 = Conv1D(filters=64, padding='causal',dilation_rate=2,kernel_size=2, strides=1, activation='tanh')  # for input1

    conv1out = conv1(input1)
    # conv1out = LeakyReLU()(conv1out)

    conv3out = Conv1D(filters=128, padding='causal',dilation_rate=2,kernel_size=4, strides=1,activation='tanh')(conv1out)
    # conv3out = LeakyReLU()(conv3out)
    conv3out = Flatten()(conv3out)

    # lstm1out = add([conv3out,conv1out])

    # lstm1out = Flatten()(lstm1out)
    lstm1out = Dense(32)(conv3out)
    # lstm1out = BatchNormalization()(lstm1out)
    # lstm1out = Activation('relu')(lstm1out)
    lstm1out = Dropout(0.2)(lstm1out)
    lstm1out = Dense(16)(lstm1out)
    output = Dense(1,activation='sigmoid')(lstm1out)

    # X = Flatten()(X)
    # X = Dropout(0.5)(X)
    # output = Dense(1)(X)

    highway_window = config.highway_window
    # 截取近3个窗口的时间维 保留了所有的输入维度
    z = Lambda(lambda k: k[:, -highway_window:, :])(input1)
    z = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1)))(z)
    z = Lambda(lambda k: K.reshape(k, (-1, highway_window * trainX1.shape[2])))(z)
    # z = Flatten()(z)
    z = Dense(1)(z)
    # z = Lambda(lambda k: K.reshape(k, (-1, trainX1.shape[2])))(z)

    output = add([output, z])

    output = Activation('sigmoid')(output)
    output = Dense(1)(output)
    model = Model(inputs=[input1, input2], outputs=output)

    # for classification, if n_out =1, add:
    # X = Activation('sigmoid')(X)

    # for classification, if n_out > 1, add:
    # X = Activation('softmax')(X)

    # Create model

    return model


def LSTNet24(trainX1, trainX2, trainY, config):
    input1 = Input(shape=(trainX1.shape[1], trainX1.shape[2]))
    input2 = Input(shape=(trainX2.shape[1], trainX2.shape[2]))

    # conv1 = Conv1D(filters=128, kernel_size=6, strides=1, activation='relu')  # for input1
    #
    # conv1out = conv1(input1)
    #
    # conv3out = Conv1D(filters=64, kernel_size=1, strides=1)(conv1out)
    # conv3out = LeakyReLU()(conv3out)
    # conv4out = Conv1D(filters=64, kernel_size=3, strides=1)(conv1out)
    # conv4out = LeakyReLU()(conv4out)
    # conv5out = Conv1D(filters=64, kernel_size=4, strides=1)(conv1out)
    # conv5out = LeakyReLU()(conv5out)
    #
    # lstm1out = Concatenate(axis=1)([conv3out, conv4out, conv5out])
    # lstm1out = Bidirectional(LSTM(32, activation=None, return_sequences=True))(lstm1out)
    # lstm1out = ELU()(lstm1out)
    # lstm1out = SeqSelfAttention(attention_activation='relu', attention_type='multiplicative')(lstm1out)
    # lstm1out = Flatten()(lstm1out)
    # lstm1out = TransformerBlock(name='1',num_heads=1)(input1)
    #
    # lstm1out = TransformerBlock(name='2',num_heads=1)(input1)
    # lstm1out = TransformerBlock(name='3',num_heads=1)(input1)
    #
    # lstm1out = TransformerBlock(name='4',num_heads=1)(input1)
    x = input1
    for i in range(4):
        x = TransformerBlock(name=i,num_heads=1)(x)
    lstm1out = Flatten()(x)




    lstm1out = Dense(16)(lstm1out)
    output = Dense(1)(lstm1out)

    # highway_window = config.highway_window
    # # 截取近3个窗口的时间维 保留了所有的输入维度
    # z = Lambda(lambda k: k[:, -highway_window:, :])(input1)
    # z = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1)))(z)
    # z = Lambda(lambda k: K.reshape(k, (-1, highway_window * trainX1.shape[2])))(z)
    #
    # z = Dense(1)(z)
    #
    # output = add([output, z])
    #
    # output = Dense(1)(output)
    model = Model(inputs=[input1, input2], outputs=output)

    return model



def trainModel(trainX1,trainX2,trainY,config):
    '''
    trainX，trainY: 训练LSTM模型所需要的数据
    config:  配置文件
    '''
    model = LSTNet22(trainX1,trainX2,trainY,config)
    model.summary()

    plot_model(model,show_shapes=True)

    # adam = optimizers.Adam(lr=0.01)
    # model.compile(loss='mean_squared_error', optimizer=adam)
    model.compile(optimizer=config.optimizer, loss=config.loss_metric)

    def scheduler(epoch):
        # 每隔100个epoch，学习率减小为原来的1/10
        if epoch % 30 == 0 and epoch != 0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr * 0.1)
            print("lr changed to {}".format(lr * 0.1))
        return K.get_value(model.optimizer.lr)

    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    lr_new = LearningRateScheduler(scheduler)

    # tbCallBack = TensorBoard(log_dir="./tensormodel", histogram_freq=1, write_grads=True)
    hist = model.fit([trainX1,trainX2], trainY, epochs=config.epochs, batch_size=config.lstm_batch_size, callbacks=[lr_new], verbose=config.verbose, validation_split=0.1)

    return model,hist



#多维归一化
def NormalizeMult(data):
    normalize = np.arange(2*data.shape[1],dtype='float64')
    normalize = normalize.reshape(data.shape[1],2)
    print(normalize.shape)
    for i in range(0,data.shape[1]):
        #第i列
        list = data[:,i]
        listlow,listhigh =  np.percentile(list, [0, 100])
        normalize[i,0] = listlow
        normalize[i,1] = listhigh
        delta = listhigh - listlow
        if delta != 0:
            #第j行
            for j in range(0,data.shape[0]):
                data[j,i]  =  (data[j,i] - listlow)/delta
    return data,normalize

#多维反归一化
def FNormalizeMult(data,normalize):
    data = np.array(data)
    for i in  range(0,data.shape[1]):
        listlow =  normalize[i,0]
        listhigh = normalize[i,1]
        delta = listhigh - listlow
        if delta != 0:
            for j in range(0,data.shape[0]):
                data[j,i]  =  data[j,i]*delta + listlow
    return data


def startTrainMult(data,name,config):
    '''
    data: 多维数据
    返回训练好的模型
    '''
    data = data.iloc[:,1:]
    print(data.columns)
    yindex = data.columns.get_loc(name)
    data = np.array(data,dtype='float64')

    #数据归一化
    data, normalize = NormalizeMult(data)
    data_y = data[:,yindex]
    data_y = data_y.reshape(data_y.shape[0],1)
    print(data.shape, data_y.shape)

    #构造训练数据
    trainX1,trainX2, _ = create_dataset(data, config.n_predictions,config.skip)
    _ , _,trainY = create_dataset(data_y,config.n_predictions,config.skip)
    print("trainX Y shape is:",trainX1.shape,trainX2.shape,trainY.shape)

    if len(trainY.shape) == 1:
        trainY = trainY.reshape(-1,1)
    # 进行训练
    model,hist = trainModel(trainX1, trainX2 , trainY, config)

    return model,hist,normalize


