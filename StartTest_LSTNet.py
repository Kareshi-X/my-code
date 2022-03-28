import  pandas as pd
import  numpy as np
from  sklearn import  metrics
from  lstm.Predict_Interface import  PredictWithData
from Config import  Config
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
# import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")
pd.set_option('display.float_format', lambda x: '%.2f' % x)
# warnings.filterwarnings("ignore")



def GetRMSE(y_hat,y_test):
    sum = np.sqrt(metrics.mean_squared_error(y_test, y_hat))
    return  sum

def GetMAE(y_hat,y_test):
    sum = metrics.mean_absolute_error(y_test, y_hat)
    return  sum

def GetMAPE(y_hat,y_test):
    sum = np.mean(np.abs((y_hat - y_test) / y_test)) * 100
    return sum

def GetMAPE_Order(y_hat,y_test):
    #删除y_test 为0元素
    zero_index = np.where(y_test == 0)
    y_hat = np.delete(y_hat,zero_index[0])
    y_test = np.delete(y_test,zero_index[0])
    sum = np.mean(np.abs((y_hat - y_test) / y_test)) * 100
    return sum

def model_performance_sc_plot(predictions, labels, title):
    # Get min and max values of the predictions and labels.
    min_val = max(max(predictions), max(labels))
    max_val = min(min(predictions), min(labels))
    # Create dataframe with predicitons and labels.
    performance_df = pd.DataFrame({"Label":labels})
    performance_df["Prediction"] = predictions
    # Plot data
    sns.jointplot(y="Label", x="Prediction", data=performance_df, kind="reg", height=7)
#     plt.plot([min_val, max_val], [min_val, max_val], 'm--')
    plt.title(title, fontsize=9)
    plt.show()
config = Config()
print(config)
path = config.multpath
data = pd.read_csv("./pollution.csv")
data = data.drop(['wnd_dir'], axis = 1)
#选取后20%
data = data.iloc[int(0.8*data.shape[0]):,:]
print("长度为",data.shape[0])
name = config.dimname

normalize = np.load(config.multpath+name+"22.npy")
loadmodelname = config.multpath+name+"22.h5"

y_hat,y_test = PredictWithData(data,name,loadmodelname,normalize,config)
y_hat = np.array(y_hat,dtype='float64')
for i in range(len(y_hat)):
    if y_hat[i] < 0:
        y_hat[i]=0

y_test = np.array(y_test,dtype='float64').reshape(-1,)

index = np.where(y_test>500)
print(index)
y_test_1 = []
y_hat_1 = []
for i in range(len(index)):
    y_test_1.append(y_test[index[i]])
    y_hat_1.append(y_hat[index[i]])

y_hat_1 = np.array(y_hat_1).reshape(1,-1)

y_test_1 = np.array(y_test_1).reshape(1,-1)

print("MSEwei",mean_squared_error(y_hat_1,y_test_1))
print("RMSEwei",GetRMSE(y_hat_1,y_test_1))
print("MAEwei",GetMAE(y_hat_1,y_test_1))
print("MAPEwei",GetMAPE_Order(y_hat_1,y_test_1))


test_mse = mean_squared_error(y_hat,y_test)

model_performance_sc_plot(y_hat[:-1].reshape(-1,),y_test[:-1].reshape(-1,), 'deep_model_test2')
print('test_mse:',test_mse)

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 15,
}

plt.figure(figsize=(22,10))
plt.xlabel('Samples',fontsize=22)
plt.ylabel('Value',fontsize=22)
# plt.plot(y_hat,label='y_hat',color='red',linewidth=3)
plt.plot(y_test,label='True',color='deepskyblue',linewidth=3,alpha=0.9)
plt.legend(fontsize=22)
plt.title('test_performance',fontsize=22)
plt.show()
plt.savefig('test_Performance.svg',format = 'svg')

# data1 = pd.read_csv("./pollution.csv")
# train = data1['pollution'][:int(0.8*data1.shape[0])]
# test = data1['pollution'][int(0.8*data1.shape[0]):]
# train.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14,label='train')
# test.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14,label='test')
# plt.legend()
# plt.show()


plt.legend()
print("RMSE为",GetRMSE(y_hat,y_test))
print("MAE为",GetMAE(y_hat,y_test))
#print("MAPE为",GetMAPE(y_hat,y_test))
print("MAPE为",GetMAPE_Order(y_hat,y_test))

r2_score = 1- mean_squared_error(y_test,y_hat)/ np.var(y_test)
print("R2_score为:", r2_score)

def smape(y_true, y_pred):

    tmp = y_pred + y_true
    zero_index = np.where(tmp == 0)
    y_hat = np.delete(y_pred,zero_index[0])
    y_test = np.delete(y_true,zero_index[0])
    # sum = np.mean(np.abs((y_hat - y_test) / y_test)) * 100
    return 2.0 * np.mean(np.abs(y_hat - y_test) / (np.abs(y_hat) + np.abs(y_test))) * 100
print("SMAPE为",smape(y_hat,y_test))
np.save(config.multpath+name+"y_hat22.npy",y_hat)
np.save(config.multpath+name+"y_test22.npy",y_test)
print("结束")


shops = ["Xgboost","Stacking", "CNN+LSTM", "Multi-CNN","Linear joint", "Attention model"]
Model = ['Xgboost','Catboost','Random forest','Lightgbm','Lasso','KNN']
# MAE = [12.2725, 12.2722, 12.0968, 11.9761]
MSE = [686.8072,540.0656,533.8638,538.4547,526.1152,520.4917]
cof = [ 0.07554803,  0.57206358, -0.19896984,  0.45589764,  0.11251956,
       -0.00587036]
# RMSE = [26.2070,23.2393, 23.1055, 23.2046, 22.8492]
# MAPE = [22.6970,22.1077, 23.2151, 21.7028, 21.3736]
# SMAPE = [20.5089,20.0129, 20.7461, 21.7028, 19.7320]


# 创建分组柱状图，需要自己控制x轴坐标
xticks = np.arange(len(shops))

# fig, ax = plt.subplots(figsize=(10, 7),linestyle='--',linewidth=3,marker='^',makersize=20)
# # 所有门店第一种产品的销量，注意控制柱子的宽度，这里选择0.25
# ax.bar(xticks, MSE, width=0.5, label="MSE", color='lightskyblue')
# # 所有门店第二种产品的销量，通过微调x轴坐标来调整新增柱子的位置
# # ax.bar(xticks + 0.25, MAPE, width=0.25, label="MAPE", color='coral')
# # # 所有门店第三种产品的销量，继续微调x轴坐标调整新增柱子的位置
# # ax.bar(xticks + 0.5, SMAPE, width=0.25, label="SMAPE", color='lightgreen')
#
#
#
# ax.set_title("Model Performance", fontsize=15)
# ax.set_xlabel("Models")
# ax.set_ylabel("Predict Error")
# ax.legend(prop=font1)


# 最后调整x轴标签的位置
# ax.set_xticks(xticks)
# ax.set_xticklabels(shops)

sns.despine()
#
# for a,b in zip(xticks,MSE):
#     plt.text(a, b, '%.4f' % b, ha='center', va= 'bottom',fontsize=12)
# plt.show()
# plt.savefig('Modelcoef.svg',format='svg')

plt.rc('font',family='Times New Roman')
plt.ylabel('Predict Error',fontdict={'family' : 'Times New Roman',  'size'   : 16})
plt.xlabel('Model',fontdict={'family' : 'Times New Roman', 'size'   : 16})

plt.xticks(fontproperties = 'Times New Roman', size = 10)
plt.yticks(fontproperties = 'Times New Roman', size = 10)

plt.plot(shops,MSE,color='cyan',alpha=0.8,linestyle='--',linewidth=3,marker='^', markersize=10,label='MSE')
plt.legend(loc='best')
plt.show()
plt.savefig('performance.svg',format='svg')
