#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from crData import create_dataset
import scipy.fftpack as spfft 

def extData(step_th, sen):
    print('Current director:')
    print (os.getcwd())
    current_dir = os.path.dirname(os.path.abspath("__file__"))
    try:
        TOT_EX_PATH = os.path.join(current_dir + '\\dataset', '[Done] awair-r2_17936_202075_2020712_linear_interpolation.csv')
        df = pd.read_csv(TOT_EX_PATH, encoding ='utf-8', index_col=0)
        df_tot = df
        list_sen = df.keys()
    except:
        print('"Error", Check it')
    print("Filename:", TOT_EX_PATH)
    train_len = 0.7
    test_len = 1-train_len
    scaler = MinMaxScaler(feature_range=(0, 1))

    # raw data 정의
    raw_data = np.array(list(df_tot[sen][10000:55000]))

    # raw data 내 train 데이터 분리
    raw_train_data = raw_data[ : int(train_len*len(raw_data))].reshape(-1,1)

    # raw data 내 test 데이터 분리
    raw_test_data = raw_data[ int(train_len*len(raw_data)): ].reshape(-1,1)

    # raw data를 normalization 
    norm_data = scaler.fit_transform(raw_data.reshape(-1,1))

    # train data의 normalized data
    train_data_scale = norm_data[ : int(train_len*len(raw_data))]

    # test data의 normalized data
    test_data_scale = norm_data[ int(train_len*len(raw_data)): ]

    # train data의 CS 인풋으로
    CS_train_data_scale = spfft.dct(train_data_scale, norm='ortho')

    # test data의 CS 인풋
    CS_test_data_scale = spfft.dct(test_data_scale, norm='ortho')
    
    # 누락 구간 (전송주기조절 구간)의 집합 =  [60, 120, 180, 240, 300, 360] 
    # 주기조절 시, 10분 간격으로 조정. 그러므로 총 60분까지 주기조절이 가능. 
    max_period = 6
    period_set = [60*i for i in range(1,max_period+1)]
    
    input_length = 360

    # 입력 데이터를 연속적으로 고려하지 않음. step_th 만큼 window 가 건너 뛰어서 움직임. 
    # 즉  현재 input 데이터 기준 10분 뒤 데이터를 항상 고려
    # step_th = 60

    # CS로 누락 구간 복원 진행 
    # CS로 누락 구간 복원을 위하여, 데이터셋을 만드는 함수 create_dataset
    
    
    input_trainX_dic = {}
    input_testX_dic = {}

    CS_trainX_dic = {}
    CS_trainY_dic = {}
    CS_testX_dic = {}
    CS_testY_dic = {}

    Dense_trainX_dic = {} 
    Dense_testX_dic = {}

    for k in period_set:
        print("***********************************************************************************************")
        print("MIssing interval: ", k)

        print("Train: ")
        input_trainX, CS_trainX, CS_trainY, error_list_1, Dense_trainX = create_dataset(CS_train_data_scale, input_length, k, step_th)
        print("Test: ")
        input_testX, CS_testX, CS_testY, error_list_2, Dense_testX = create_dataset(CS_test_data_scale, input_length, k, step_th)


        # 실제 알고리즘에 들어가는 input 데이터 
        input_trainX_dic[k] = input_trainX
        input_testX_dic[k] = input_testX

        # CS 데이터 복원시 사용되는 데이터 집합
        CS_trainX_dic[k] = CS_trainX
        CS_trainY_dic[k] = CS_trainY

        CS_testX_dic[k] = CS_testX
        CS_testY_dic[k] = CS_testY

        # Dense layer에 같이 들어가는 missing 값  
        Dense_trainX_dic[k] = np.array(Dense_trainX).astype('float32')
        Dense_testX_dic[k] = np.array(Dense_testX).astype('float32')
        print('Dataset_Size => input_train_data_size : ({}), CS_train_data_size: ({})'.format(input_trainX_dic[k].shape, CS_trainX_dic[k].shape))
        print('Dataset_Size => input_test_data_size : ({}), CS_test_data_size: ({})'.format(input_testX_dic[k].shape, CS_testX_dic[k].shape))

    
    return input_trainX_dic, input_testX_dic, CS_trainX_dic, CS_trainY_dic, CS_testX_dic, CS_testY_dic, Dense_trainX_dic, Dense_testX_dic, scaler      


# In[ ]:





# In[ ]:




