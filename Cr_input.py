#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from crData import create_dataset
import scipy.fftpack as spfft 


# In[26]:


#######################    Data Loading ########################

def Cr_input():
    
    max_period = 6
    period_set = [60*i for i in range(1,max_period+1)]
    
    file_name_MSE = 'error_LSTM\\'
    file_name_CS = 'LSTM_CS_data\\'

    input_trainY_MSE_dic = {} 
    input_trainY_RMSE_dic = {}

    input_testY_MSE_dic = {}
    input_testY_RMSE_dic = {}

    CS_impu_train_dic = {}
    CS_impu_test_dic = {}

    for i in period_set:
        # CS 복원 데이터의 MSE 및 RMSE 값. train 과 test에서 따로 저장 
        input_trainY_MSE_dic[i] = pd.read_csv(file_name_MSE + 'train_MSE_LSTM_error_' + str(i) + '.csv', index_col = 0 )
        input_trainY_RMSE_dic[i] = pd.read_csv(file_name_MSE + 'train_RMSE_LSTM_error_' + str(i) + '.csv', index_col = 0)
        input_testY_MSE_dic[i] = pd.read_csv(file_name_MSE + 'test_MSE_LSTM_error_' + str(i) + '.csv', index_col = 0)
        input_testY_RMSE_dic[i] = pd.read_csv(file_name_MSE + 'test_RMSE_LSTM_error_' + str(i) + '.csv',index_col = 0)

        # CS 복원 데이터셋
        CS_impu_train_dic[i] = pd.read_csv(file_name_CS + 'train_CS_' + str(i) + '.csv', index_col = 0)
        CS_impu_test_dic[i] = pd.read_csv(file_name_CS + 'test_CS_' + str(i) + '.csv', index_col = 0)


    ############################### Loadinng  데이터에 적용 ################################# 
    xOut_LSTM_dic = {}
    xTest_LSTM_dic = {}
    for num, i in enumerate(period_set[:-1]):

        # xOut_LSTM_dic -> CS가 복원한 Train 데이터를 누락 구간에 상관 없이 동일하게 지정
        # xTest_LSTM_dic -> CS가 복원한 Test 데이터를 누락 구간에 상관 없이 동일하게 지정

        xOut_LSTM_dic[i] = np.expand_dims(np.array(CS_impu_train_dic[i])[: -(len(period_set) - num - 1)], axis=2)
        xTest_LSTM_dic[i] = np.expand_dims(np.array(CS_impu_test_dic[i])[: -(len(period_set) - num - 1)], axis=2)
    xOut_LSTM_dic[period_set[-1]] = np.expand_dims(np.array(CS_impu_train_dic[period_set[-1]]), axis=2)
    xTest_LSTM_dic[period_set[-1]] = np.expand_dims(np.array(CS_impu_test_dic[period_set[-1]]),axis = 2)
    
    return input_trainY_MSE_dic, input_trainY_RMSE_dic, input_testY_MSE_dic, input_testY_RMSE_dic, CS_impu_train_dic, CS_impu_test_dic


# In[28]:





# In[ ]:





# In[ ]:




