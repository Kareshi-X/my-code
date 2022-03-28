import os
import  pandas as pd
import  numpy as np
from lstm.LSTNet_Interface import  startTrainMult
from Config import Config


from keras.utils  import plot_model

config = Config()

path = config.multpath
print(path)
isExists = os.path.exists(path)
if not isExists:
    os.makedirs(path)


data = pd.read_csv("./pollution.csv")
data = data.drop(['wnd_dir'], axis = 1)
# data = data.query('pollution<900')
data = data.iloc[:int(0.8*data.shape[0]),:]
print("长度为",data.shape[0])
name = config.dimname



model,hist,normalize = startTrainMult(data,name,config)
# model.plot_model()
#在某些情况下模型无法直接保存 需要保存权重
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'],label='loss')
plt.plot(hist.history['val_loss'],label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.title('train_loss')
plt.show()
plt.savefig('loss.svg',format='svg')

model.save_weights(config.multpath+name+"22.h5")
np.save(config.multpath+name+"22.npy",normalize)


