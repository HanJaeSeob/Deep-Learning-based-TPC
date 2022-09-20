#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
import tensorflow as tf
import keras
import pandas as pd
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib import pyplot as plt
import time
import copy
import math
import shutil
from sklearn.metrics import mean_squared_error
from tqdm.notebook import tqdm
import matplotlib as mpl
# from keras.utils import plot_model


# ## 실제 CS가 복원하는 데이터의 누락복원 정확도 도출
# ### - input data 정의
# #### (1) CS_trainY_dic: CS가 복원해야하는 데이터
# #### (2) xOut_LSTM_dic: CS가 복원한 데이터
# #### (3) input_trainX_dic[360] : 실제 input train data 패턴 -> 패턴에 따라서 분류해야함
# #### (4) input_testX_dic[360] : 실제 input test data 패턴-> 패턴에 따라서 분류해야함
# #### (5) input_trainY_RMSE_dic: CS의 실제 train 데이터 복원 정확도를 RMSE로 지정
# #### (6) xOut_LSTM_dic:  CS가 복원한 Train 데이터를 누락 구간에 상관 없이 동일하게 지정
# #### (7)  xTest_LSTM_dic:  CS가 복원한 Test 데이터를 누락 구간에 상관 없이 동일하게 지정

# # 1. Imputation Accuracy Model 학습
# ## 1) Bi-LSTM 모델 학습
# ### (1) 학습 데이터 형성
# ### (2) 학습 구문 생성
# ### (3) 학습된 모델 로딩
# ### (4) 성능 추출 구문 생성

# In[2]:


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# ## (1) Bi-LSTM 학습 데이터 로딩

# In[3]:


input_length = 360
max_period = 6
period_set = [60*i for i in range(1,max_period+1)]
from Cr_input import Cr_input
from extData import extData
st = 6
input_trainX_dic, input_testX_dic, CS_trainX_dic, CS_trainY_dic, CS_testX_dic, CS_testY_dic, Dense_trainX_dic, Dense_testX_dic, scaler = extData(st, 'ico2')
# input_trainY_MSE_dic, input_trainY_RMSE_dic, input_testY_MSE_dic, input_testY_RMSE_dic, CS_impu_train_dic, CS_impu_test_dic = Cr_input()


# ## (2) 복원 데이터셋 만들기

# In[4]:


# from impu_CS import CNN_gen_dataY, Impu_gen_dataY
# input_trainY_MSE_dic, input_trainY_RMSE_dic, CS_impu_train_dic =  CNN_gen_dataY(period_set, input_length, CS_trainX_dic, CS_trainY_dic, scaler)
# # Generation CNN_testY data
# input_testY_MSE_dic, input_testY_RMSE_dic, CS_impu_test_dic =  CNN_gen_dataY(period_set, input_length, CS_testX_dic, CS_testY_dic, scaler)
# Li_input_testY_MSE_dic, Li_input_testY_RMSE_dic, Li_impu_test_dic =  Impu_gen_dataY(period_set, input_length, CS_testX_dic, CS_testY_dic,'linear',scaler)
# from crFolder import createFolder, dict_format
# trainY_MSE, trainY_RMSE, testY_MSE, testY_RMSE = {}, {}, {}, {}
# trainY_MSE = dict_format(input_trainY_MSE_dic, trainY_MSE)
# trainY_RMSE = dict_format(input_trainY_RMSE_dic, trainY_RMSE)
# testY_MSE = dict_format(input_testY_MSE_dic, testY_MSE)
# testY_RMSE = dict_format(input_testY_RMSE_dic, testY_RMSE)

#### File Save ####### 
# for i in CS_impu_train_dic.keys():
#     file_name_train = 'CS_train_' + str(i) + '.csv'
#     file_name_test = 'CS_test_' + str(i) + '.csv'
#     file_path_train = os.path.join(file_path, file_name_train)
#     file_path_test = os.path.join(file_path, file_name_test)
    
#     pd.DataFrame(CS_impu_train_dic[i]).to_csv(file_path_train)
#     pd.DataFrame(CS_impu_test_dic[i]).to_csv(file_path_test)
    
# pd.DataFrame(trainY_MSE).to_csv(os.path.join(file_path, 'trainY_MSE.csv'))
# pd.DataFrame(trainY_RMSE).to_csv(os.path.join(file_path, 'trainY_RMSE.csv'))
# pd.DataFrame(testY_MSE).to_csv(os.path.join(file_path, 'testY_MSE.csv'))
# pd.DataFrame(testY_RMSE).to_csv(os.path.join(file_path, 'testY_RMSE.csv'))


# ## (3) 복원 데이터셋 저장 및 불러오기

# In[5]:


from crFolder import createFolder, dict_format
data_folder_name = 'error_data_storage'

cur_dir = os.getcwd()
file_path = os.path.join(cur_dir, data_folder_name)
file_name_train = 'trainY_RMSE.csv'
file_name_test = 'testY_RMSE.csv'
trainY_RMSE = pd.read_csv(os.path.join(file_path, file_name_train),  index_col = 0)
testY_RMSE = pd.read_csv(os.path.join(file_path, file_name_test),  index_col = 0)


# In[6]:


LSTM_trainY = {}
LSTM_testY = {}
for i in trainY_RMSE.keys():
    temp_trainY = pd.read_csv(os.path.join(file_path, 'CS_train_'+ str(i) + '.csv'), index_col = 0)
    temp_testY = pd.read_csv(os.path.join(file_path, 'CS_test_'+ str(i) + '.csv'), index_col = 0)
    LSTM_trainY[i] = temp_trainY
    LSTM_testY[i] = temp_testY


# ## (4) LSTM model Define For Training

# In[6]:


############################################ FOr CNN - LSTM ###############################################################
from keras import optimizers
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, Conv1D, Conv2D, GlobalMaxPooling1D, Flatten, Concatenate, MaxPooling1D, Lambda, BatchNormalization, Bidirectional, Activation
from keras.models import Model, Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

model_input = Input(shape = (input_length, 1))

#################################################### LSTM #################################################################
xLstm_1 = Bidirectional(LSTM(5, return_sequences=True))(model_input)
xLstm_2 = Bidirectional(LSTM(5, return_sequences=True))(xLstm_1)
xLstm_3 = Bidirectional(LSTM(5, return_sequences=True))(xLstm_2)
xLstm_4 = Bidirectional(LSTM(5, return_sequences=True))(xLstm_3)
xLstm_5 = Bidirectional(LSTM(5, return_sequences=True))(xLstm_4)
xLstm_6 = Bidirectional(LSTM(5, return_sequences=True))(xLstm_5)



xLstm_1 = Flatten()(xLstm_1)
xLstm_2 = Flatten()(xLstm_2)
xLstm_3 = Flatten()(xLstm_3)
xLstm_4 = Flatten()(xLstm_4)
xLstm_5 = Flatten()(xLstm_5)
xLstm_6 = Flatten()(xLstm_6)

xLstm_1 = Dense(10, activation = 'relu')(xLstm_1)
xLstm_2 = Dense(10, activation = 'relu')(xLstm_2)
xLstm_3 = Dense(10, activation = 'relu')(xLstm_3)
xLstm_4 = Dense(10, activation = 'relu')(xLstm_4)
xLstm_5 = Dense(10, activation = 'relu')(xLstm_5)
xLstm_6 = Dense(10, activation = 'relu')(xLstm_6)



# xLstm_2 = Flatten()(xLstm_2)
# xLstm_1 = Flatten()(xLstm_1)

xIm_1 = Dense(60)(xLstm_1)
xIm_2 = Dense(120)(xLstm_2)
xIm_3 = Dense(180)(xLstm_3)
xIm_4 = Dense(240)(xLstm_4)
xIm_5 = Dense(300)(xLstm_5)
xIm_6 = Dense(360)(xLstm_6)


xTr_1 = Dense(60)(xLstm_1)
xTr_2 = Dense(120)(xLstm_2)
xTr_3 = Dense(180)(xLstm_3)
xTr_4 = Dense(240)(xLstm_4)
xTr_5 = Dense(300)(xLstm_5)
xTr_6 = Dense(360)(xLstm_6)


# sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

###################################### For CNN learning ########################################################################
BiLSTM_model = Model(inputs=model_input, outputs= [xTr_1, xTr_2, xTr_3, xTr_4, xTr_5, xTr_6, xIm_1, xIm_2, xIm_3, xIm_4, xIm_5, xIm_6])

BiLSTM_model.compile(loss="mse", optimizer="adam", metrics=["acc"])
BiLSTM_model.summary()


# In[7]:


input_trainX = input_trainX_dic[360]
input_testX = input_testX_dic[360]
trainY = [np.array(CS_trainY_dic[k][:len(LSTM_trainY['360'])]).reshape(len(LSTM_trainY['360']), k) for k in period_set]
valY = [np.array(CS_testY_dic[k][:len(LSTM_testY['360'])]).reshape(len(LSTM_testY['360']), k) for k in period_set]
for k in period_set:
    trainY.append(np.array(LSTM_trainY[str(k)][:len(LSTM_trainY[str(360)])]))
    valY.append(np.array(LSTM_testY[str(k)][:len(LSTM_testY[str(360)])]))

# hist = BiLSTM_model.fit([input_trainX], [trainY], batch_size = batch_size, epochs=epoch, shuffle= False, validation_data=([input_testX], [valY]), verbose=1, callbacks=[cp_callback])


# ## (5) Bi-LSTM Learning run code

# In[9]:


epoch = 1200
batch_size = 32
# checkpoint_path = "co2_training_1/LSTM_LSTM_cp-{epoch:04d}.ckpt"   -> 이전 epoch으로 실행
checkpoint_path = "Model_save/J4-BiLSTM_cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, period=1)
# es = EarlyStopping(monitor='val_loss', verbose=1, patience=50)
BiLSTM_model.save_weights(checkpoint_path.format(epoch=0))


# ## GPU 확인
# + J2 : 2개의 Bidirectional LSTM model을 정의하고, 각 LSTM 모델에서 모든 구간을 예측
# + J3 : 6개의 Bidirectional LSTM model을 정의하고, 각 모델에서는 각 구간의 데이터 값을 예측
# + J4 : 6개의 Bidirectional LSTM model을 정의하고, 각 모델에서의 output feautre 수를 조절함

# In[ ]:


with tf.device('/CPU:0'):
    hist = BiLSTM_model.fit(input_trainX, trainY, batch_size = batch_size, epochs=epoch, shuffle= False, validation_data=(input_testX, valY), verbose=1, callbacks=[cp_callback])


# + J2 Case
# Epoch 00481: saving model to Model_save/J1-BiLSTM_cp-0481.ckpt - 0.0608
# + J4 Case
# Epoch 00471: saving model to Model_save/J4-BiLSTM_cp-0471.ckpt - 0.0603
# + J5 Case 
# Epoch 00510: saving model to Model_save/J4-BiLSTM_cp-0510.ckpt - 0.0578
# Epoch 00602: saving model to Model_save/J4-BiLSTM_cp-0602.ckpt - 0.0582

# ## (6) Bi-LSTM 모델 최적의 파라미터값 Loading

# In[8]:


optimal_LSTM_ep = "510"
BiLSTM_model.load_weights("Model_save/J4-BiLSTM_cp-0" + optimal_LSTM_ep +".ckpt")
print("Loaded model from disk")


# ## (7) Bi-LSTM 결과 뽑기

# In[11]:


# # Save the LSTM imputation OuTput (Learnging CS output)
# LSTM_Dout_dic = {}
# LSTM_Iout_dic = {}
# for i in period_set:
#     LSTM_Dout_dic[i] = []
#     LSTM_Iout_dic[i] = [] 
# for i in tqdm(range(int(input_testX.shape[0]))):
#     with tf.device('/GPU:0'):
#         temp_out = BiLSTM_model.predict([input_testX[i].reshape(1,360,1)])
#     for k,i in enumerate(period_set):
#         LSTM_Dout_dic[i].append(temp_out[k].reshape(-1))
#         LSTM_Iout_dic[i].append(temp_out[k+6].reshape(-1))
        
# from crFolder import createFolder, dict_format
# current_dir = os.getcwd()
# lstm_folder_dir = current_dir + '\\lstm_out\\' 
# createFolder(lstm_folder_dir)
# for i in period_set:
#     file_name_LSTMD = 'lstm_Dout_train_v' + str(i) + '.csv'
#     pd.DataFrame(LSTM_Dout_dic[i]).T.to_csv(lstm_folder_dir + file_name_LSTMD)
#     file_name_LSTMI = 'lstm_Iout_train_v' + str(i) + '.csv'
#     pd.DataFrame(LSTM_Iout_dic[i]).T.to_csv(lstm_folder_dir + file_name_LSTMI)
#     print(['Save Done LSTM Output of ' + str(i)])


# ## (8) LSTM 결과 Graph
# + 실제 데이터와 유사도 추출 Graph
# + 복원 데이터와 유사도 추출 Graph

# In[11]:


ml = 240
ml_ind = 900
plt.plot(CS_testY_dic[ml][ml_ind])
plt.plot(LSTM_Dout_dic[ml][ml_ind])
#plt.plot(LSTM_testY[str(ml)].iloc[ml_ind])
#plt.plot(LSTM_Iout_dic[ml][ml_ind])


# ## (9) LSTM model Loading For training CNN

# In[14]:


############################################ FOr CNN - LSTM ###############################################################
from keras import optimizers
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, Conv1D, Conv2D, GlobalMaxPooling1D, Flatten, Concatenate, MaxPooling1D, Lambda, BatchNormalization, Bidirectional, Activation
from keras.models import Model, Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

model_input = Input(shape = (input_length, 1))

#################################################### LSTM #################################################################
xLstm_1 = Bidirectional(LSTM(5, return_sequences=True))(model_input)
xLstm_2 = Bidirectional(LSTM(5, return_sequences=True))(xLstm_1)
xLstm_3 = Bidirectional(LSTM(5, return_sequences=True))(xLstm_2)
xLstm_4 = Bidirectional(LSTM(5, return_sequences=True))(xLstm_3)
xLstm_5 = Bidirectional(LSTM(5, return_sequences=True))(xLstm_4)
xLstm_6 = Bidirectional(LSTM(5, return_sequences=True))(xLstm_5)



xLstm_1 = Flatten()(xLstm_1)
xLstm_2 = Flatten()(xLstm_2)
xLstm_3 = Flatten()(xLstm_3)
xLstm_4 = Flatten()(xLstm_4)
xLstm_5 = Flatten()(xLstm_5)
xLstm_6 = Flatten()(xLstm_6)

xLstm_1 = Dense(10, activation = 'relu')(xLstm_1)
xLstm_2 = Dense(10, activation = 'relu')(xLstm_2)
xLstm_3 = Dense(10, activation = 'relu')(xLstm_3)
xLstm_4 = Dense(10, activation = 'relu')(xLstm_4)
xLstm_5 = Dense(10, activation = 'relu')(xLstm_5)
xLstm_6 = Dense(10, activation = 'relu')(xLstm_6)



# xLstm_2 = Flatten()(xLstm_2)
# xLstm_1 = Flatten()(xLstm_1)

xIm_1 = Dense(60)(xLstm_1)
xIm_2 = Dense(120)(xLstm_2)
xIm_3 = Dense(180)(xLstm_3)
xIm_4 = Dense(240)(xLstm_4)
xIm_5 = Dense(300)(xLstm_5)
xIm_6 = Dense(360)(xLstm_6)


xTr_1 = Dense(60)(xLstm_1)
xTr_2 = Dense(120)(xLstm_2)
xTr_3 = Dense(180)(xLstm_3)
xTr_4 = Dense(240)(xLstm_4)
xTr_5 = Dense(300)(xLstm_5)
xTr_6 = Dense(360)(xLstm_6)


# sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

###################################### For CNN learning ########################################################################
BiLSTM_model = Model(inputs=model_input, outputs= [xLstm_1, xLstm_2, xLstm_3, xLstm_4, xLstm_5, xLstm_6, xTr_1, xTr_2, xTr_3, xTr_4, xTr_5, xTr_6, xIm_1, xIm_2, xIm_3, xIm_4, xIm_5, xIm_6])

BiLSTM_model.compile(loss="mse", optimizer="adam", metrics=["acc"])
BiLSTM_model.summary()

optimal_LSTM_ep = "510"
BiLSTM_model.load_weights("Model_save/J4-BiLSTM_cp-0" + optimal_LSTM_ep +".ckpt")
print("Loaded model from disk")


# ## (10) Input 및 Test Data에서 누락 복원 정확도 저장

# In[15]:


# 각 인풋 데이터 마다 RMSE 6개 값(누락구간)을 저장한 array 
erTrain = np.zeros((len(LSTM_trainY[str(360)]), len(period_set)))
for i in range(len(LSTM_trainY[str(360)])):
    temp_list = []
    for j in period_set:
        temp_list.append(trainY_RMSE[str(j)].iloc[i])
        np_list = np.array(temp_list)
    erTrain[i] = np_list
    
    
# 각 인풋 데이터 마다 RMSE 6개 값(누락구간)을 저장한 array 
erTest = np.zeros((len(testY_RMSE[str(360)]), len(period_set)))
for i in range(len(testY_RMSE[str(360)])):
    temp_list = []
    for j in period_set:
        temp_list.append(testY_RMSE[str(j)].iloc[i])
        np_list = np.array(temp_list)
    erTest[i] = np_list


# ## (11) 구간별 평균 누락복원 정확도 추출

# In[16]:


avg_train = []
avg_test = []
for i in period_set:
    avg_train.append(np.mean(np.array(trainY_RMSE[str(i)])))
    avg_test.append(np.mean(np.array(testY_RMSE[str(i)])))
plt.plot(avg_train)


# ## 2) CNN 모델 학습
# ### (1) 데이터 로딩
# ### (2) 학습 구문 생성
# ### (3) 학습된 모델 로딩
# ### (4) 성능 추출 구문 생성

# ## (1) CNN 모델 학습을 위한 데이터 생성

# In[11]:


Dense_trainX_input = []
Dense_testX_input = []
cur_input_trainX = input_trainX_dic[360]
cur_input_testX = input_testX_dic[360]

input_trainY = []
input_testY = []

for num, i in enumerate(period_set[:-1]):
#     print(num)
    Dense_trainX_input = Dense_trainX_input + list(Dense_trainX_dic[i][:len(Dense_trainX_dic[360])])
    Dense_testX_input = Dense_testX_input + list(Dense_testX_dic[i][:len(Dense_testX_dic[360])])
    
    input_trainY = input_trainY + list(np.array(trainY_RMSE[str(i)]))
    input_testY = input_testY + list(np.array(testY_RMSE[str(i)]))
    
    cur_input_trainX = np.vstack((cur_input_trainX, input_trainX_dic[360]))
    cur_input_testX = np.vstack((cur_input_testX, input_testX_dic[360]))
    
Dense_trainX_input = Dense_trainX_input + list(Dense_trainX_dic[period_set[-1]])
Dense_testX_input = Dense_testX_input + list(Dense_testX_dic[period_set[-1]])

input_trainY = input_trainY + list(np.array(trainY_RMSE[str(period_set[-1])]).reshape(-1))
input_testY = input_testY + list(np.array(testY_RMSE[str(period_set[-1])]).reshape(-1))

input_Dense_trainX = np.array(Dense_trainX_input).astype('float32')
input_Dense_testX = np.array(Dense_testX_input).astype('float32')

input_trainX = cur_input_trainX
input_testX = cur_input_testX


# ## (2) Multi-head CNN model Define

# In[17]:


################################### MIssing interval 의 정보 density를 높임 앞쪽으로 #################################################### 
##################################### input의 dimension을 하나 더 추가해서 1d cnn에 인가하기 ###########################################
num_filters = 128
Dense_layer_1 = 100
Dense_layer_2 = 20
# model_input = Input(shape = (input_length, 1))

model_input = Input(shape = (360, 1))

miss_input = Input(shape= (1,))

#################################################### LSTM #################################################################

[xLstm_1, xLstm_2, xLstm_3, xLstm_4, xLstm_5, xLstm_6, xTr_1, xTr_2, xTr_3, xTr_4, xTr_5, xTr_6, xIm_1, xIm_2, xIm_3, xIm_4, xIm_5, xIm_6] = BiLSTM_model(model_input)

Period_1 = Dense(10, activation='sigmoid')(miss_input)
Period_2 = Dense(10, activation='sigmoid')(miss_input)
Period_3 = Dense(10, activation='sigmoid')(miss_input)
Period_4 = Dense(10, activation='sigmoid')(miss_input)
Period_5 = Dense(10, activation='sigmoid')(miss_input)
Period_6 = Dense(10, activation='sigmoid')(miss_input)

xLstm_1 = Lambda(lambda x : tf.add(x[0], x[1]))([xLstm_1, Period_1])
xLstm_2 = Lambda(lambda x : tf.add(x[0], x[1]))([xLstm_2, Period_2])
xLstm_3 = Lambda(lambda x : tf.add(x[0], x[1]))([xLstm_3, Period_3])
xLstm_4 = Lambda(lambda x : tf.add(x[0], x[1]))([xLstm_4, Period_4])
xLstm_5 = Lambda(lambda x : tf.add(x[0], x[1]))([xLstm_5, Period_5])
xLstm_6 = Lambda(lambda x : tf.add(x[0], x[1]))([xLstm_6, Period_6])




Period_Info = Dense(1, activation='sigmoid')(miss_input)
Period_Info = Dense(1, activation='sigmoid')(Period_Info)

BiLSTM_model.trainable = True
for k, layer in enumerate(BiLSTM_model.layers):
    if k in range(19):
        layer.trainable = True
    else:
        layer.trainable = False


#################################################### CNN #################################################################

missing_input_copy = Lambda(lambda x : tf.keras.backend.repeat(x, 360))(Period_Info)

cnn_input = Lambda(lambda x : tf.add(x[0], x[1]))([model_input, missing_input_copy])


kernel_blocks = [10, 15, 30, 60, 90]
conv_blocks = []
for kernel in kernel_blocks:
    conv = Conv1D(filters = num_filters,
                         kernel_size = kernel,
                         padding = "valid",
                         activation = "relu",
                         strides = 2)(cnn_input)
    conv = Conv1D(filters = num_filters,
                         kernel_size = kernel,
                         padding = "valid",
                         activation = "relu",
                         strides = 2)(cnn_input)
    conv = MaxPooling1D(pool_size=2, strides=1, padding='valid')(conv)
    conv = Lambda(lambda x : x[ : , -135:, : ])(conv)
#     conv = Cropping1D(cropping=1)
#     conv = GlobalMaxPooling1D()(conv)
    conv_blocks.append(conv)
x_CNN_1 = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
# x_CNN_1 = Lambda(lambda x : tf.expand_dims(x,2))(x_CNN_1)
x_CNN_1 = Conv1D(filters = num_filters, kernel_size = 3, padding = "valid", strides = 1)(x_CNN_1)
# x_CNN_1 = BatchNormalization()(x_CNN_1)
x_CNN_1 = Activation('relu')(x_CNN_1)
x_CNN_1 = Conv1D(filters = num_filters, kernel_size = 3, padding = "valid", strides = 1)(x_CNN_1)
x_CNN_1 = BatchNormalization()(x_CNN_1)
x_CNN_1 = Activation('relu')(x_CNN_1)
x_CNN_1 = MaxPooling1D()(x_CNN_1)

x_CNN_1 = Conv1D(filters = num_filters, kernel_size = 3, padding = "valid", strides = 1)(x_CNN_1)
# x_CNN_1 = BatchNormalization()(x_CNN_1)
x_CNN_1 = Activation('relu')(x_CNN_1)
x_CNN_1 = Conv1D(filters = num_filters, kernel_size = 3, padding = "valid", strides = 1)(x_CNN_1)
x_CNN_1 = BatchNormalization()(x_CNN_1)
x_CNN_1 = Activation('relu')(x_CNN_1)
x_CNN_1 = MaxPooling1D()(x_CNN_1)

x_CNN_1 = Conv1D(filters = num_filters, kernel_size = 3, padding = "valid", strides = 1)(x_CNN_1)
# x_CNN_1 = BatchNormalization()(x_CNN_1)
x_CNN_1 = Activation('relu')(x_CNN_1)
x_CNN_1 = Conv1D(filters = num_filters, kernel_size = 3, padding = "valid", strides = 1)(x_CNN_1)
x_CNN_1 = BatchNormalization()(x_CNN_1)
x_CNN_1 = Activation('relu')(x_CNN_1)
x_CNN_1 = MaxPooling1D()(x_CNN_1)


# 기존의 커널 사이즈 : x_CNN_1 = Conv1D(filters = num_filters, kernel_size = 5, padding = "valid", strides = 2)(x_CNN_1)


x_CNN_1 = Flatten()(x_CNN_1)

feature_out = Concatenate()([xLstm_1, xLstm_2, xLstm_3, xLstm_4, xLstm_5, xLstm_6, x_CNN_1])
feature_out = Lambda(lambda x : tf.expand_dims(x,2))(feature_out)


Dense_1 = Flatten()(feature_out)
Dense_1 = Concatenate()([Dense_1, Period_Info])
Dense_1 = Dense(Dense_layer_1)(Dense_1)
# Dense_1 = BatchNormalization()(Dense_1)
Dense_1 = Activation('relu')(Dense_1)
# Dense_2 = Dense(Dense_layer_2, activation='relu')(Dense_1)
Dense_2 = Dense(Dense_layer_2)(Dense_1)

# output_error = Dense(1, activation='relu')(Dense_2)
output_error = Dense(1)(Dense_2)

CNN_LSTM_model_v3 = Model(inputs=[model_input, miss_input], outputs=[output_error, missing_input_copy, Period_1, Period_2, Period_3, Period_4, Period_5, Period_6])
CNN_LSTM_model_v3.compile(loss="mse",  optimizer='adam', metrics=["acc"])

CNN_LSTM_model_v3.summary()


# ## (3) Multi-head CNN Model Training

# In[18]:


############################### CNN - LSTM model v1 학습 !!!!!!!! ########################################################
############################## CNN_LSTM_4 -> 1d cnn input 조절 ##############################
epoch = 2000
batch_size = 32
checkpoint_path = "Model_save/JS3-CNN_cp-{epoch:01d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback_CNN = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, period=1)
CNN_LSTM_model_v3.save_weights(checkpoint_path.format(epoch=0))


# In[20]:


# 최적값 : co2_training_1/CNN_LSTM_Period_cp-36.ckpt
with tf.device('/GPU:0'):
    hist_cnn = CNN_LSTM_model_v3.fit([input_trainX, input_Dense_trainX/60], np.array(input_trainY), batch_size = batch_size, epochs=epoch, shuffle= False, validation_data=([input_testX, input_Dense_testX/60], np.array(input_testY)), verbose=1, callbacks=[cp_callback_CNN])


# ## (4) 최적의 CNN 파라미터 Loading

# In[18]:


optimal_CNN_ep = "17"
CNN_LSTM_model_v3.load_weights("Model_save/JS3-CNN_cp-" + optimal_CNN_ep +".ckpt")
print("Loaded model from disk")


# In[19]:


input_Dense_testX[0].reshape(1,1)/60


# In[20]:


th_list = [0.5107752, 0.5136615, 0.51653844, 0.51935077, 0.5220478, 0.5245875]


# In[ ]:





# In[21]:


plt.plot(input_testX[200] + th_list[0])
plt.plot(input_testX[200] + th_list[5])


# In[ ]:





# In[49]:


CNN_LSTM_model_v3.predict([input_testX[120].reshape(1,360,1), np.array([4])])


# ## (5) CNN model output 저장

# In[30]:


# CNN 결과의 아웃풋을 저장함
CNN_out_dic = {}
result_out = []
result_out_dic = {}

for i in period_set:
    CNN_out_dic[i] = []
with tf.device('/GPU:0'):    
    for i in tqdm(range(input_testX.shape[0])):
        temp_out = CNN_LSTM_model_v3.predict([input_testX[i].reshape(1,360,1), input_Dense_testX[i].reshape(1,1)/60])
        result_out.append(temp_out.reshape(-1))
#     pd.DataFrame(result_out).to_csv('CNN_output\\CNN_output_result.csv')
#     pd.DataFrame(input_testY).to_csv('CNN_output\\True_values.csv')


# In[41]:


result_dic = {}
result_out = []
# data_index = 200
np_period_value = np.linspace(1,6,100)
with tf.device('/GPU:0'):
    for data_index in tqdm(range(0,int(len(input_testX)/6), 20)):
        result_out = []
        for i in tqdm(range(len(np_period_value))):
            temp_out = CNN_LSTM_model_v3.predict([input_testX[data_index].reshape(1,360,1), np_period_value[i].reshape(1,1)])
            result_out.append(temp_out.reshape(-1)[0])
        result_dic[data_index] = result_out


# In[83]:


#     pd.DataFrame(result_out).to_csv('CNN_output\\CNN_output_result.csv')
#     pd.DataFrame(input_testY).to_csv('CNN_output\\True_values.csv')



pd.read_csv('CNN_output\\CNN_output_result.csv', in_column =0)


# In[100]:


ind_fixed = 24
quality_value = result_dic[list(result_dic.keys())[ind_fixed]]


# In[101]:


plt.plot(quality_value)


# In[ ]:





# In[22]:


def energy_fun(x):
    a,b = -1/6, 1
    y = a*x+b
    return y

# quality_value = np.zeros(len(result_out))
for j in np_period_value:
    energy_value = [energy_fun(j) for j in np_period_value]


# In[103]:


quality_value = np.array(quality_value)
plt.plot(quality_value)


# In[ ]:





# In[108]:


alpha = 0.04
object_val = np.zeros(len(quality_value))
for i in range(len(quality_value)):
    object_val[i] = max(quality_value[i]*alpha, np.array(energy_value)[i])


# In[109]:


fontdict={'fontname': 'Times New Roman',
     'fontsize': 20,
     'style': 'italic', # 'oblique'
      'fontweight': 'bold'}  # 'heavy', 'light', 'ultrabold', 'ultralight'


plt.style.use('classic')
plt.figure(num=1,dpi=100,facecolor='white')
plt.grid()

# x,y axis 폰트 속성 설정
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15
plt.rcParams['font.style'] = 'italic'


plt.ylabel('Objective value', **fontdict)
plt.xlabel('Period', **fontdict)
plt.plot(object_val, lw=4, marker = 'o',markevery=10, markerfacecolor=(1,1, 1), markersize=10 , color = 'brown', label = 'Objective function')
plt.legend(loc='upper right', frameon=True)


# In[110]:


import celluloid


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[111]:


alpha = 0.09
plt.plot(quality_value*alpha)
plt.plot(energy_value)


# In[ ]:


def energy_fun(x):
    a,b = -1/6, 1
    y = a*x+b
    return y
def quality_fun(u,x, alpha):
    with tf.device('/CPU:0'):
        y = alpha*CNN_LSTM_model_v3.predict([u.reshape(1,360,1), np.array(x).reshape(1,1)])
    return y
def objective_fun(energy_fun, quality_fun, u,x, alpha):
    y = max(energy_fun(x), quality_fun(u,x, alpha))
    return y


# In[ ]:


def energy_fun(x):
    a,b = -1/6, 1
    y = a*x+b
    return y

opt_energy_performance = np.zeros(6)
for i in range(1,7):
    opt_energy_performance[i-1] = energy_fun(i)


# In[ ]:





# In[ ]:





# In[9]:


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


# In[ ]:





# In[ ]:





# In[7]:


# # CNN 결과의 아웃풋을 저장함
# CNN_out_dic = {}
# result_out = []
# result_out_dic = {}

# for i in period_set:
#     CNN_out_dic[i] = []
# with tf.device('/GPU:0'):    
#     for i in tqdm(range(input_testX.shape[0])):
#         temp_out = CNN_LSTM_model_v3.predict([input_testX[i].reshape(1,360,1), input_Dense_testX[i].reshape(1,1)/60])
#         result_out.append(temp_out.reshape(-1))
#     pd.DataFrame(result_out).to_csv('CNN_output\\CNN_output_result.csv')
#     pd.DataFrame(input_testY).to_csv('CNN_output\\True_values.csv')


# In[8]:


result_out = pd.read_csv('CNN_output\\CNN_output_result.csv', index_col = 0)
input_testY = pd.read_csv('CNN_output\\True_values.csv', index_col = 0)
result_out = list(result_out['0'])
input_testY = list(input_testY['0'])


# In[25]:


(31.55+36.25)/2


# In[ ]:





# In[24]:


index = 5
start = int(index*len(result_out)/6)
end = int((index+1)*len(result_out)/6)
# plt.plot(result_out[start:end])
# plt.plot(input_testY[start:end])

fontdict={'fontname': 'Times New Roman',
     'fontsize': 20,
     'style': 'italic', # 'oblique'
      'fontweight': 'bold'}  # 'heavy', 'light', 'ultrabold', 'ultralight'


plt.style.use('classic')
plt.figure(num=1,dpi=100,facecolor='white')
plt.grid()

# x,y axis 폰트 속성 설정
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15
plt.rcParams['font.style'] = 'italic'

plt.xlim(0,2100)
plt.ylabel('RMSE', **fontdict)
plt.xlabel('Index of Time', **fontdict)
plt.plot(input_testY[start:end], lw=1.5, color = 'brown', label = 'Ground Truth')
plt.plot(result_out[start:end], lw=3.5,  color = 'blue', label = 'PIEU')
# plt.plot(moving_average(input_testY[start:end], 120), lw=2.5,  color = 'cyan', label = 'Moving Average')
# plt.plot(result_out[start:end], lw=1.5, marker = 'o',markevery=50, markerfacecolor=(1,1, 1), markersize=10 , color = 'blue', label = 'PAI')
# plt.plot(input_testY[start:end], lw=1.5, marker = '*',markevery=50, markerfacecolor=(1,1, 1), markersize=10 , color = 'brown', label = 'Ground Truth')

plt.legend(loc='upper right', frameon=True)


# In[ ]:





# In[ ]:





# ## (6) CNN prediction 결과 추출

# In[10]:


result_out = pd.read_csv('CNN_output\\CNN_output_result.csv', index_col = 0)
input_testY = pd.read_csv('CNN_output\\True_values.csv', index_col = 0)
result_out = list(result_out['0'])
input_testY = list(input_testY['0'])

plt.plot(result_out)
plt.plot(input_testY)

def MAE(realD, compD):
    MAE_out = np.mean(abs(np.array(realD).reshape(-1)-np.array(compD).reshape(-1)))
    return MAE_out

def RMSE(realD, compD):
    RMSE_out = np.sqrt(np.mean((np.array(realD).reshape(-1)-np.array(compD).reshape(-1))**2))
    return RMSE_out
interval = int(len(result_out)/6)
metric = np.zeros((6,2))
for j in range(6):
    realD = [input_testY[interval*j:interval*(j+1)][i] for i in solList]
    compD = [result_out[interval*j:interval*(j+1)][i] for i in solList]
    
    a = round(RMSE(realD, compD),2)
    b = round(MAE(realD, compD),2)
    print('RMSE:{}, MAE:{}'.format(a,b))
    metric[j][0] = a
    metric[j][1] = b


# In[ ]:





# In[9]:


disp_alpha = '0.005'
solList_read = pd.read_csv(disp_alpha + '_solList.csv', index_col = 0)
solList = np.array(solList_read).reshape(-1)


# In[ ]:





# # 2. 제안하는 방법의 주기값 추출
# ## 1) 에너지 함수 정의

# In[14]:


def energy_fun(x):
    a,b = -1/6, 1
    y = a*x+b
    return y

opt_energy_performance = np.zeros(6)
for i in range(1,7):
    opt_energy_performance[i-1] = energy_fun(i)


# ## 2) 주기 추출 시, 사용되는 CPU , GPU 사용량 추출

# In[ ]:


import time
import threading
import os
import psutil
import GPUtil

list_GPU_usage, list_GPU_unusage, list_GPU_usage_percent = [], [], []
list_total_cpu_usage_percent, list_total_memory_usage, list_instance_memory_usage = [], [], []
# total_cpu_usage_percent, total_memory_usage, cpu_usage, cpu_usage_percent, instance_memory_usage = _check_usage_of_cpu_and_memory()
# GPU_usage, GPU_unusage, GPU_usage_percent = _check_usage_of_gpu_and_memory()


def _check_usage_of_cpu_and_memory():

    global list_total_cpu_usage_percent, list_total_memory_usage, list_cpu_usage, list_cpu_usage_percent, list_instance_memory_usage
    
    pid = os.getpid()
    py  = psutil.Process(pid)
    
    # 전체 컴퓨터에서 소비하는 cpu와 memory 사용량 정보
    total_cpu_usage_percent = psutil.cpu_percent()
    memory_usage_dict = dict(psutil.virtual_memory()._asdict())
    total_memory_usage = memory_usage_dict['used']
    
    # 현재 python instance에서 사용하는 CPU 정보
#     cpu_usage   = os.popen("ps aux | grep " + str(pid) + " | grep -v grep | awk '{print $3}'").read()
#     cpu_usage   = cpu_usage.replace("\n","")
    
#     cpu_usage_percent = py.cpu_percent()
    
    # 현재 python instance에서 사용하는 메모리 정보
    instance_memory_usage  = py.memory_info()[0] /2.**30   # GiB(기가바이트)
    
    # Memory total : 34287910912
    
    
    list_total_cpu_usage_percent.append(total_cpu_usage_percent)
    list_total_memory_usage.append(total_memory_usage)
#     list_cpu_usage.append(cpu_usage)
#     list_cpu_usage_percent.append(cpu_usage_percent)
    list_instance_memory_usage.append(instance_memory_usage)
    
#     print("cpu usage\t\t:", cpu_usage, "%")
#     print("memory usage\t\t:", instance_memory_usage, "%")
    
def _check_usage_of_gpu_and_memory():
    
    global list_GPU_usage, list_GPU_unusage, list_GPU_usage_percent
    
    GPU_usage = GPUtil.getGPUs()[0].memoryUsed
    GPU_unusage = GPUtil.getGPUs()[0].memoryFree
    GPU_usage_percent = GPUtil.getGPUs()[0].memoryUtil*100
    GPU_memory_total = GPUtil.getGPUs()[0].memoryTotal
    # GPU_memory_total = 8192.0
    
    list_GPU_usage.append(GPU_usage)
    list_GPU_unusage.append(GPU_unusage)
    list_GPU_usage_percent.append(GPU_usage_percent)


class Monitor(threading.Thread):
    
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.suspend = False
        self.delay = delay # Time between calls to GPUtil
        self.daemon = True
        self.start()

    def run(self):
        while not self.stopped:
            while self.suspend:
                time.sleep(5)
            
            _check_usage_of_cpu_and_memory()
            _check_usage_of_gpu_and_memory()
            
            time.sleep(self.delay)
                
    def Suspend(self):
        self.suspend = True
         
    def Resume(self):
        self.suspend = False
            
    def stop(self):
        self.stopped = True


# In[ ]:





# In[37]:


u = input_testX[0]
x = 1
0.05*CNN_LSTM_model_v3.predict([u.reshape(1,360,1), np.array(x).reshape(1,1)])[0]


# ## 3) 주기 추출 코드

# In[38]:


import time
import threading

def energy_fun(x):
    a,b = -1/6, 1
    y = a*x+b
    return y
def quality_fun(u,x, alpha):
    with tf.device('/CPU:0'):
        y = alpha*CNN_LSTM_model_v3.predict([u.reshape(1,360,1), np.array(x).reshape(1,1)])[0]
    return y
def objective_fun(energy_fun, quality_fun, u,x, alpha):
    y = max(energy_fun(x), quality_fun(u,x, alpha))
    return y


t_min = 1
t_max = 6
l_iter = 100000
eps = 0.00001
t_feas = 0.5

input_testX = cur_input_testX
input_testX = input_testX[0:int(input_testX.shape[0]/6)]

# feasibility 문제를 푸는 알고리즘
# t_feas가 주어지면, 그것에 해당하는 energy function의 값과 quality function의 x값을 각각 구하는 bisection_feasibility 함수
# output으로는 t_feas에 해당하는 energy function의 x값과 quality function의 x값, 그리고 해당 x값이 일치 유무를 나타내는 result로 구분(feasible)

def bisection_feasibility(function_e, function_q, u, t_feas, alpha):
    l = 0
    t_min = 1
    t_max = 6
    eps = 0.00001
    l_iter = 10000
    t_max_e, t_min_e = t_max, t_min
    t_max_q, t_min_q = t_max, t_min
    
    while((t_max_q-t_min_q>eps) & (l < l_iter)):
        t_q = (t_min_q + t_max_q)/2
        if function_q(u, t_q, alpha) < t_feas:
            t_min_q = t_q
        else:
            t_max_q = t_q
        l = l + 1
    t_quality = t_q
    
    t_energy =  -6*(t_feas - 1)

    if abs(max(energy_fun(t_energy), quality_fun(u, t_energy, alpha)) - t_feas) < 0.0005:
        t_hat = t_energy
        result = 'feasible'
    elif abs(max(energy_fun(t_quality), quality_fun(u, t_quality, alpha)) - t_feas)  <0.0005:
        t_hat = t_quality
        result = 'feasible'
    else:

        t_hat = min(t_energy, t_quality)
        result = 'infeasible'

    return t_energy, t_quality, result, t_hat



# alpha: quality function과 energy function의 반영비율을 나타내는 parameter
# opt_X_dic: 주기 값을 저장하는 dictionary (alpha에 따라서)
# energy_dic: energy performance을 저장하는 dictionary (alpha에 따라서)
# quality_dic: quality performance을 저장하는 dictionary (alpha에 따라서)
# alpha = 10
prop_dic = {}
energy_dic = {}
quality_dic = {}

alp_ran = [0.05]

# alp_ran = [0.1]

energy_performance = []
quality_performance = []

list_GPU_usage, list_GPU_unusage, list_GPU_usage_percent = [], [], []
list_total_cpu_usage_percent, list_total_memory_usage, list_instance_memory_usage = [], [], []


t_min = 1
t_max = 6
opt_X_result = []
eps = 0.0001
with tf.device('/CPU:0'):
    for alp in tqdm(alp_ran):
#         monitor = Monitor(30)
#         time.sleep(60*60*1)
#         list_total_cpu_usage_percent.append('END')
        opt_X_result = []
        for i in tqdm(range(int(input_testX.shape[0]))):
            # bisection searching 하기 위하여, solution의 상한값과 하한값을 지정함 
            lowVal = min(energy_fun(t_max), quality_fun(input_testX[i], t_min, alp))
            highVal = min(energy_fun(t_min), quality_fun(input_testX[i],t_max, alp))

            l = 0

            while((highVal - lowVal > eps) & (l < l_iter)):
                t_feas = (lowVal + highVal)/2
                eps_1 = 0.0001
                # feasibility를 check하는 단계
                t_energy, t_quality, result, t_hat = bisection_feasibility(energy_fun, quality_fun, input_testX[i], t_feas, alp)

                if result == 'feasible':
                    highVal = t_feas
                else:
                    lowVal = t_feas

                l = l + 1
                x_hat = t_hat
            print('==========================================================================================================================')
            print('Final result:', result)
            print('Final t_feasibility:', t_feas)

            u = input_testX[i]
            ceil_x_hat = math.ceil(x_hat)
            floor_x_hat = math.floor(x_hat)

            sol_list = [ceil_x_hat, floor_x_hat]
            opt_ind = np.argmin([objective_fun(energy_fun, quality_fun, u, ceil_x_hat, alp), objective_fun(energy_fun, quality_fun, u, floor_x_hat, alp)])
            opt_x = sol_list[opt_ind]
            print("Optimal X:", opt_x)

            energy_performance.append(energy_fun(opt_x))
            quality_performance.append(quality_fun(input_testX[i], opt_x, alp))
            opt_X_result.append(opt_x)


        prop_dic[alp] = opt_X_result
        energy_dic[alp] = energy_performance
        quality_dic[alp] = quality_performance
    
# 추출 및 에너지 등의 추출된 값 저장
pd.DataFrame(prop_dic).to_csv('20220209_P.csv')
pd.DataFrame(energy_dic).to_csv('20220209_E.csv')
pd.DataFrame(quality_dic).to_csv('20220209_Q.csv')

# GPU 및 CPU 사용량 저장
# monitor.stop()
# GPU_df = pd.DataFrame([list_GPU_usage, list_GPU_unusage, list_GPU_usage_percent]).T
# GPU_df.columns = ['usage', 'unusgae', 'usage_percent']
# GPU_df.to_csv('GPU_usage.csv')
# CPU_df = pd.DataFrame([list_total_cpu_usage_percent, list_total_memory_usage, list_instance_memory_usage]).T
# CPU_df.columns = ['usage_percent', 'memory_usage', 'instance_memory_usage']
# CPU_df.to_csv('CPU_usage.csv')


# ## 4) CPU 및 GPU 사용량 결과 추출

# In[2]:


GPU_df = pd.read_csv('GPU_usage.csv', index_col = 0)
CPU_df = pd.read_csv('CPU_usage.csv', index_col = 0)


# In[3]:


ind_k = 0
for k, i in enumerate(CPU_df['usage_percent']):
    if i == "END":
        ind_k = k
print('index_k:', ind_k)


# In[4]:


plt.plot(CPU_df['usage_percent'][ind_k-60:ind_k].astype('float32').to_list())
plt.plot(CPU_df['usage_percent'][ind_k+2:ind_k+1+61].astype('float32').to_list())


# In[5]:


CPU_usage_percent = np.mean(CPU_df['usage_percent'][ind_k-60:ind_k].astype('float32'))
CPU_unusage_percent = np.mean(CPU_df['usage_percent'][ind_k+2:ind_k+1+61].astype('float32'))

GPU_usage_percent = np.mean(GPU_df['usage_percent'][ind_k-60:ind_k].astype('float32'))
GPU_unusage_percent = np.mean(GPU_df['usage_percent'][ind_k+2:ind_k+1+61].astype('float32'))

print('CPU usage & unusage: {}, {}'.format(CPU_usage_percent, CPU_unusage_percent))
print('GPU usage & unusage: {}, {}'.format(GPU_usage_percent, GPU_unusage_percent))


# In[6]:


index_instance_list = []
index_usage_list = []

for i in range(len(df_cpu_usage)):
    index_instance_list.append('CPU')
    index_usage_list.append('Idle mode')Comp
for i in range(len(df_gpu_usage)):
    index_instance_list.append('GPU')
    index_usage_list.append('Idle mode')
for i in range(len(df_cpu_usage)):
    index_instance_list.append('CPU')
    index_usage_list.append('Running algorithm')
for i in range(len(df_gpu_usage)):
    index_instance_list.append('GPU')
    index_usage_list.append('Running algorithm')
len(index_instance_list)
len(index_usage_list)


# In[ ]:


# df_cpu_usage = pd.DataFrame([CPU_df['usage_percent'][ind_k-60:ind_k].to_list(), CPU_df['usage_percent'][ind_k+2:ind_k+1+61]]).T
# df_gpu_usage = pd.DataFrame([GPU_df['usage_percent'][ind_k-60:ind_k].to_list(), GPU_df['usage_percent'][ind_k+2:ind_k+1+61]]).T
# len(df_gpu_usage)


# In[7]:


df_usage = pd.DataFrame(CPU_df['usage_percent'][ind_k-60:ind_k].to_list() + GPU_df['usage_percent'][ind_k-60:ind_k].to_list() + CPU_df['usage_percent'][ind_k+2:ind_k+1+61].to_list() + GPU_df['usage_percent'][ind_k+2:ind_k+1+61].to_list())
df_usage.columns = ['percent']
df_usage['instance'] = index_instance_list
df_usage['Usage type'] = index_usage_list
df_usage['percent'] = df_usage['percent'].astype('float32')


# In[150]:


usage_bar = sns.barplot(x='instance', y= 'percent', hue = 'Usage type',  data=df_usage)
plt.title("CPU/GPU usage (%)")
usage_bar.set_xlabel('Instance Type', fontsize = 15)
usage_bar.set_ylabel('Usage (%)', fontsize = 15)
plt.show()


# In[ ]:





# # 3. Linear Approx 기반 데이터 주기 조절
# + https://tariat.tistory.com/819

# In[153]:


from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans


# ## 1) Elbow Method 사용 및 Silhouette 계수 출력

# In[154]:


# model = KMeans()
# visualizer = KElbowVisualizer(model, k=(1,10))
# visualizer.fit(input_trainX_dic[360].reshape(5069,360))


# In[155]:


# model = KMeans(2)
# visualizer = SilhouetteVisualizer(model)

# visualizer.fit(input_trainX_dic[360].reshape(5069,360))    # Fit the data to the visualizer
# visualizer.poof() 


# In[ ]:


plt.plot(avg_train, marker='o', linewidth=5)
plt.xlabel('Missing Interval', fontsize = 20)
plt.xticks(np.arange(0,6),[f"{i*10} min".format(i) for i in range(1,7)])
plt.ylabel('RMSE', fontsize = 20)
plt.show()


# ## 2) 최적의 K값 선정

# In[160]:


sse = []

for i in tqdm(range(1,11)):
    km = KMeans(n_clusters=i,algorithm='auto', random_state=42)
    km.fit(input_trainX_dic[360].reshape(5069,360))
    sse.append(km.inertia_)

plt.plot(range(1,11), sse, marker='o', linewidth=4)
plt.xlabel('K', fontsize = 20)
plt.ylabel('SSE', fontsize = 20) # SSE is defined as the sum of the squared distance between centroid and each member of the cluster.
plt.show()


# In[166]:


avg_train = []
avg_test = []
for i in period_set:
    avg_train.append(np.mean(np.array(trainY_RMSE[str(i)])))
    avg_test.append(np.mean(np.array(testY_RMSE[str(i)])))
# plt.plot(avg_train)


# ## 3) 선정된 K값 기반 클러스터 형성

# In[167]:


#########  Linear 전송 주기값 계산 및 quality 값과 energy 값 구하기 (test 데이터에서 진행) ############
### K-means clustering centriod 추출 ### 
start = time.time()
cluster_num = 2 
from tslearn.clustering import TimeSeriesKMeans
model = TimeSeriesKMeans(n_clusters=cluster_num, max_iter=50)
km = model.fit(input_trainX_dic[360])
run_time = time.time() - start
print('Run_time:', run_time)


# In[168]:


# cluster label 추출 For each train and test data
predicted_train_labels = km.predict(input_trainX_dic[360])
predicted_test_labels = km.predict(input_testX_dic[360])


# In[169]:


# test 데이터 내 clustering 진행 -> label 추출
import collections
labels = []
sizes = []
# label들의 갯수를 나타내는 sizes list
for i in range(cluster_num):
    labels.append("cluster_"+str(i))
    sizes.append(collections.Counter(predicted_test_labels)[i])
print(sizes)

index = {}
for i in range(cluster_num):
    index[i] = []

for i in range(len(input_testX_dic[360])):
    if predicted_test_labels[i] == 0:
        index[0].append(i)
    elif predicted_test_labels[i] == 1:
        index[1].append(i)


# In[170]:


# RMSE_Dtype: train dataset 내 각 데이터에서 missing 구간을 달리하면서 (6까지) CS 기반 Imputation 에러 저장 (506 x 6)
cluster = {}
for j in range(cluster_num):
    temp = np.zeros((6))
    for i in range(erTrain[index[j]].shape[0]):
        temp = temp + erTrain[index[j]][i]
    cluster[j] = np.hstack((np.array([0]),(temp/erTrain[index[j]].shape[0])))
    # cluster: 각 클러스터 별 에러의 평균값 저장 (2 x 6)


# ## 4) 클러스터 기반, Linear regression을 통한 Error 모델링

# In[171]:


# Linear regression 진행
# Train data 기반 linear regression 진행
from sklearn.linear_model import LinearRegression
cluster0_line_fitter = LinearRegression()
cluster1_line_fitter = LinearRegression()
X = np.array([i for i in range(7)]).reshape(-1,1)
y_cluster0 = cluster[0]
y_cluster1 = cluster[1]
cluster0_line_fitter.fit(X, y_cluster0)
cluster1_line_fitter.fit(X, y_cluster1)

# cluster0_line_fitter.coef_
# cluster0_line_fitter.intercept_


# In[183]:


# legend, title등 font 속성

fontdict={'fontname': 'Times New Roman',
     'fontsize': 20,
     'style': 'italic', # 'oblique'
      'fontweight': 'bold'}  # 'heavy', 'light', 'ultrabold', 'ultralight'


plt.style.use('classic')
plt.figure(num=1,dpi=100,facecolor='white')
plt.grid()

# x,y axis 폰트 속성 설정
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15
plt.rcParams['font.style'] = 'italic'


plt.legend(loc='upper left', frameon=False)
plt.title('Cluster')
plt.ylabel('Imputation Error (RMSE)', **fontdict)
plt.xlabel('Missing Interval', **fontdict)

plt.scatter([i for i in range(7)], cluster[0], alpha=0.8, marker = '^', c='green', s=100, label ='Cluster 0')
plt.scatter([i for i in range(7)], cluster[1], alpha=0.8, marker = 'o', c='red', s=100, label ='Cluster 1')
plt.plot(cluster0_line_fitter.coef_*X + cluster0_line_fitter.intercept_, color= "green", linewidth=5, linestyle="-", label="Linear regression of Cluster 0")
plt.plot(cluster1_line_fitter.coef_*X + cluster1_line_fitter.intercept_, color= "red", linewidth=5, linestyle="-", label="Linear regression of Cluster 1")
# plt.scatter([i for i in range(7)], cluster[1], marker = 'o', markerfacecolor='red', markersize=10)
plt.xlim(0,6)
plt.legend(loc='upper left', frameon=False)
plt.title('Optimum Period control', **fontdict)
# plt.ylabel('RMSE', **fontdict)


# ## 5) Linear 기반 모델의 결과값 추출 및 저장

# In[30]:


alpha_lin = 0.01
from li_per_ext import li_per_ext
li_per, li_energy, li_quality = li_per_ext(alpha_lin, input_testX_dic, erTest, cluster0_line_fitter, cluster1_line_fitter, predicted_test_labels)
linear_output_folder ="linear_output"
createFolder(linear_output_folder)
li_per.to_csv(os.path.join(linear_output_folder, str(alpha_lin) + '_li_per.csv'))
li_energy.to_csv(os.path.join(linear_output_folder, str(alpha_lin) + '_li_energy.csv'))
li_quality.to_csv(os.path.join(linear_output_folder, str(alpha_lin) + '_li_quality.csv'))


# In[20]:





# In[ ]:




