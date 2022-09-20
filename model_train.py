
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


input_length = 360
max_period = 6
period_set = [60*i for i in range(1,max_period+1)]
from Cr_input import Cr_input
from extData import extData
st = 6
input_trainX_dic, input_testX_dic, CS_trainX_dic, CS_trainY_dic, CS_testX_dic, CS_testY_dic, Dense_trainX_dic, Dense_testX_dic, scaler = extData(st, 'ico2')

from crFolder import createFolder, dict_format
data_folder_name = 'error_data_storage'

cur_dir = os.getcwd()
file_path = os.path.join(cur_dir, data_folder_name)
file_name_train = 'trainY_RMSE.csv'
file_name_test = 'testY_RMSE.csv'
trainY_RMSE = pd.read_csv(os.path.join(file_path, file_name_train),  index_col = 0)
testY_RMSE = pd.read_csv(os.path.join(file_path, file_name_test),  index_col = 0)


LSTM_trainY = {}
LSTM_testY = {}
for i in trainY_RMSE.keys():
    temp_trainY = pd.read_csv(os.path.join(file_path, 'CS_train_'+ str(i) + '.csv'), index_col = 0)
    temp_testY = pd.read_csv(os.path.join(file_path, 'CS_test_'+ str(i) + '.csv'), index_col = 0)
    LSTM_trainY[i] = temp_trainY
    LSTM_testY[i] = temp_testY



input_trainX = input_trainX_dic[360]
input_testX = input_testX_dic[360]
trainY = [np.array(CS_trainY_dic[k][:len(LSTM_trainY['360'])]).reshape(len(LSTM_trainY['360']), k) for k in period_set]
valY = [np.array(CS_testY_dic[k][:len(LSTM_testY['360'])]).reshape(len(LSTM_testY['360']), k) for k in period_set]
for k in period_set:
    trainY.append(np.array(LSTM_trainY[str(k)][:len(LSTM_trainY[str(360)])]))
    valY.append(np.array(LSTM_testY[str(k)][:len(LSTM_testY[str(360)])]))



erTrain = np.zeros((len(LSTM_trainY[str(360)]), len(period_set)))
for i in range(len(LSTM_trainY[str(360)])):
    temp_list = []
    for j in period_set:
        temp_list.append(trainY_RMSE[str(j)].iloc[i])
        np_list = np.array(temp_list)
    erTrain[i] = np_list
    
    
erTest = np.zeros((len(testY_RMSE[str(360)]), len(period_set)))
for i in range(len(testY_RMSE[str(360)])):
    temp_list = []
    for j in period_set:
        temp_list.append(testY_RMSE[str(j)].iloc[i])
        np_list = np.array(temp_list)
    erTest[i] = np_list





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



BiLSTM_model = Model(inputs=model_input, outputs= [xLstm_1, xLstm_2, xLstm_3, xLstm_4, xLstm_5, xLstm_6, xTr_1, xTr_2, xTr_3, xTr_4, xTr_5, xTr_6, xIm_1, xIm_2, xIm_3, xIm_4, xIm_5, xIm_6])



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



epoch = 2000
batch_size = 32
checkpoint_path = "Model_save/JS3-CNN_cp-{epoch:01d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback_CNN = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, period=1)
CNN_LSTM_model_v3.save_weights(checkpoint_path.format(epoch=0))


with tf.device('/GPU:0'):
    hist_cnn = CNN_LSTM_model_v3.fit([input_trainX, input_Dense_trainX/60], np.array(input_trainY), batch_size = batch_size, epochs=epoch, shuffle= False, validation_data=([input_testX, input_Dense_testX/60], np.array(input_testY)), verbose=1, callbacks=[cp_callback_CNN])


