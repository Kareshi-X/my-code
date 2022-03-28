import numpy as np
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from lstm.LSTNet_Interface import create_dataset,LSTNet,LSTNet2,LSTNet3,LSTNet4,LSTNet5,LSTNet6,LSTNet7,LSTNet8,LSTNet9,LSTNet10,LSTNet11,LSTNet12,LSTNet13,LSTNet14,LSTNet15,LSTNet16,LSTNet17,LSTNet18,LSTNet19,LSTNet20,LSTNet21,LSTNet22,LSTNet23,LSTNet24
from keras.utils import plot_model
from keras_sequential_ascii import keras2ascii
# from quiver_engine import server
#设定为自增长
configtf = tf.ConfigProto()
configtf.gpu_options.allow_growth=True
session = tf.Session(config=configtf)
KTF.set_session(session)

#指定传入该单维的最大最小值
def FNormalize_Single(data,norm):
    listlow = norm[0]
    listhigh = norm[1]
    delta = listhigh - listlow
    if delta != 0:
        for i in range(len(data)):
            data[i,0] =  data[i,0]*delta + listlow
    return  data

#使用训练数据的归一化
def NormalizeMultUseData(data,normalize):
    data = np.array(data)
    for i in range(0, data.shape[1]):
        #第i列
        listlow = normalize[i, 0]
        listhigh = normalize[i, 1]
        delta = listhigh - listlow
        if delta != 0:
            #第j行
            for j in range(0,data.shape[0]):
                data[j,i]  =  (data[j,i] - listlow)/delta
    return  data

def PredictWithData(data,name,modelname,normalize,config):

    data = data.iloc[:, 1:]
    print(data.columns)
    yindex = data.columns.get_loc(name)
    data = np.array(data, dtype='float64')

    #归一化
    data = NormalizeMultUseData(data, normalize)
    data_y = data[:, yindex]
    data_y = data_y.reshape(data_y.shape[0], 1)

    testX1,testX2, _ = create_dataset(data, config.n_predictions,config.skip)
    _ , _,testY = create_dataset(data_y,config.n_predictions,config.skip)
    print("testX Y shape is:",testX1.shape, testX2.shape,testY.shape)
    if len(testY.shape) == 1:
        testY = testY.reshape(-1,1)

    model = LSTNet22(testX1, testX2, testY, config)
    # keras2ascii(model)

    # server.launch(model, classes=['cat', 'dog'], input_folder='./imgs')

    # with open(r'json_model.json', 'w',encoding='utf-8') as file:
    #     file.write(model.to_json())
    #
    model.load_weights(modelname)
    plot_model(model,show_shapes=True)

    print('加载权重成功')
    model.summary()

    #加载模型
    y_hat =  model.predict([testX1,testX2])
    print('预测值为',y_hat)

    #反归一化
    testY = FNormalize_Single(testY, normalize[yindex,])
    y_hat = FNormalize_Single(y_hat, normalize[yindex,])
    return  y_hat,testY