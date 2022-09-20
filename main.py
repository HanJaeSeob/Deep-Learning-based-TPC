
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



from Cr_input import Cr_input
from extData import extData




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

###################################### For CNN learning ########################################################################
BiLSTM_model = Model(inputs=model_input, outputs= [xLstm_1, xLstm_2, xLstm_3, xLstm_4, xLstm_5, xLstm_6, xTr_1, xTr_2, xTr_3, xTr_4, xTr_5, xTr_6, xIm_1, xIm_2, xIm_3, xIm_4, xIm_5, xIm_6])

BiLSTM_model.compile(loss="mse", optimizer="adam", metrics=["acc"])

num_filters = 128
Dense_layer_1 = 100
Dense_layer_2 = 20

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


if __name__ == "__main__":


    prop_dic = {}
    energy_dic = {}
    quality_dic = {}
    st = 6
    input_trainX_dic, input_testX_dic, CS_trainX_dic, CS_trainY_dic, CS_testX_dic, CS_testY_dic, Dense_trainX_dic, Dense_testX_dic, scaler = extData(st, 'ico2')

    input_length = 360
    max_period = 6
    period_set = [60*i for i in range(1,max_period+1)]
    input_trainX = input_trainX_dic[360]
    input_testX = input_testX_dic[360]

    alp_ran = [0.05]

    optimal_CNN_ep = "17"
    CNN_LSTM_model_v3.load_weights("Model/JS3-CNN_cp-" + optimal_CNN_ep +".ckpt")
    print("Loaded model from disk")

    energy_performance = []
    quality_performance = []


    t_min = 1
    t_max = 6
    opt_X_result = []
    eps = 0.0001
    t_feas = 0.5

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
