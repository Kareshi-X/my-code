from keras.layers import Dense, LSTM,Conv1D,Dropout,concatenate,add,Flatten
from keras.layers.core import  Lambda,Activation
from keras.models import K,Model,Input
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
# from keras.experimental import *
import numpy as np
# from tensorflow import keras
from keras.layers import *
from keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestRegressor
from attention import Attention

from keras import regularizers

# from keras.layers import Input, Dense, LSTM, merge ,Conv1D,Dropout,Bidirectional,Multiply
from keras.models import Model

from .seq_self_attention import SeqSelfAttention
# from attention_utils import get_activations
from keras.layers import merge,add
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from keras.utils import plot_model
from .Resnet50Conv1d_BN import identity_block,convolutional_block,ResNet50


class attention_layers(Layer):
    def __init__(self, **kwargs):
        super(attention_layers, self).__init__(**kwargs)

    def build(self,inputshape):
        assert len(inputshape) == 3
        #以下是keras的自己开发工作
        self.W = self.add_weight(name='attr_weight',
                                 shape=(inputshape[1], inputshape[2]),
                                 initializer='uniform',
                                 trainable=True)
        self.b = self.add_weight(name='attr_bias',
                                 shape=(inputshape[1],),
                                 initializer='uniform',
                                 trainable=True)
        super(attention_layers, self).bulid(inputshape)

    def call(self,inputs):
        x = K.permute_dimensions(inputs, (0, 2, 1))
        a = K.softmax(K.tanh(K.dot(x, self.W) + self.b))
        outputs = K.permute_dimensions(a*x, (0, 2, 1))
        outputs = K.sum(outputs, axis=1)
        return  outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]