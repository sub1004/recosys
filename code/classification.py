#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras import layers
from keras.models import Model
from keras.layers import Input, Embedding, Dot, Add, Flatten
from keras.regularizers import l2
from keras.optimizers import SGD, Adam, rmsprop
from keras.layers import Dropout, Flatten, Conv1D
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from keras.models import load_model
import numpy as np

################ 파일 로드
books = pd.read_csv('../data/BX-Books.csv', encoding='cp1252')
books = books.drop_duplicates(subset='ISBN')
users = pd.read_csv('../data/BX-users.csv', encoding='cp1252')
ratings = pd.read_csv('../data/BX-Book-Ratings.csv', encoding='cp1252')
ratings = ratings[ratings['Book-Rating']!=0].copy()
ratings['Book-Rating'] = (ratings['Book-Rating']==10).astype(int)


import keras.backend as KB
def RMSE(y_true, y_pred):
    return KB.sqrt(KB.mean((y_true - y_pred)**2))


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+KB.epsilon()))
def recall_m(y_true, y_pred):
        true_positives = KB.sum(KB.round(KB.clip(y_true * y_pred, 0, 1)))
        possible_positives = KB.sum(KB.round(KB.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + KB.epsilon())
        return recall
def precision_m(y_true, y_pred):
        true_positives = KB.sum(KB.round(KB.clip(y_true * y_pred, 0, 1)))
        predicted_positives = KB.sum(KB.round(KB.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + KB.epsilon())
        return precision

################ 전처리
# 국가 추출
#print(users['Location'].apply(lambda x: x.split(',')[-1].strip()).unique())
users['Location'] = users['Location'].apply(lambda x: x.split(',')[-1].strip())
usa_list = ['usa', 'united state', 'united states', 'u.s.a.']
unknown_list = ['','far away...', 'here and there']
england_list = ['england','united kingdom']
spain_list = ['orense','spain']
# usa, unknown으로 처리
users['Location'] = users['Location'].replace(to_replace=usa_list, value='usa')
users['Location'] = users['Location'].replace(to_replace=unknown_list, value='unknown')

#england, spain 나라,도시 위치 오류 수정
users['Location'] = users['Location'].replace(to_replace=england_list, value='england')
users['Location'] = users['Location'].replace(to_replace=spain_list, value='spain')

# user, book, rating 테이블 병합
ratings = pd.merge(pd.merge(ratings, books, on='ISBN', how='left'), users, on='User-ID', how='left')
# 최대 나이 122로 제한
ratings.loc[ratings['Age']>120, 'Age'] = 122

# 나이, 출판일자 범주형으로 변환
ratings['Age'] = ratings['Age'].apply(lambda x: str(int(x//10*10)))
ratings['Year-Of-Publication'] = ratings['Year-Of-Publication'].apply(lambda x: str(int(x//10*10)))

# 범주형 변수를 숫자로 변환
le = {}
cols = ['ISBN','Book-Author','Publisher','Year-Of-Publication','User-ID','Location','Age']
for col in cols:
    le[col] = LabelEncoder()
    ratings[col] = le[col].fit_transform(ratings[col].astype(str))

features = ['ISBN','Book-Author','Publisher','Year-Of-Publication','User-ID','Location','Age']
#train, test = train_test_split(ratings[features + ['Book-Rating']], test_size=0.3, random_state=40)
X_train, X_test, Y_train, Y_test = train_test_split(ratings[features], 
                                                    ratings['Book-Rating'].values, test_size=0.3, random_state=11)

mu = Y_train.mean()
batch_size = 16


# In[2]:


#conv_mf
import reco_models
from keras import backend as KB
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+KB.epsilon()))
def recall_m(y_true, y_pred):
        true_positives = KB.sum(KB.round(KB.clip(y_true * y_pred, 0, 1)))
        possible_positives = KB.sum(KB.round(KB.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + KB.epsilon())
        return recall
def precision_m(y_true, y_pred):
        true_positives = KB.sum(KB.round(KB.clip(y_true * y_pred, 0, 1)))
        predicted_positives = KB.sum(KB.round(KB.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + KB.epsilon())
        return precision
def conv_mf_clf(df, K=10, reg=0.0005):
    ### Conv
    # User embedding
    user = Input(shape=(1,))            
    location = Input(shape=(1,))
    age = Input(shape=(1,))

    N = len(set(df['User-ID']))
    P_user = Embedding(N, K, embeddings_regularizer=l2(reg))(user)  
    P_user = layers.Reshape((-1,1))(P_user)
    N = len(set(df['Location']))
    P_location = Embedding(N, K, embeddings_regularizer=l2(reg))(location)
    P_location = layers.Reshape((-1,1))(P_location)
    N = len(set(df['Age']))
    P_age = Embedding(N, K, embeddings_regularizer=l2(reg))(age)
    P_age = layers.Reshape((-1,1))(P_age)

    P_concat = layers.Concatenate(axis=-1)([P_user, P_location, P_age])
    P_conv = layers.Conv1D(filters=1, kernel_size=1)(P_concat)
    P_embedding = Flatten()(P_conv)

    # Item embedding
    item = Input(shape=(1,))            
    author = Input(shape=(1,))            
    publisher = Input(shape=(1,))            
    publication_year = Input(shape=(1,))    

    N = len(set(df['ISBN']))
    Q_item = Embedding(N, K, embeddings_regularizer=l2(reg))(item)    
    Q_item = layers.Reshape((-1,1))(Q_item)
    N = len(set(df['Book-Author']))
    Q_author = Embedding(N, K, embeddings_regularizer=l2(reg))(author) 
    Q_author = layers.Reshape((-1,1))(Q_author)
    N = len(set(df['Publisher']))
    Q_publisher = Embedding(N, K, embeddings_regularizer=l2(reg))(publisher) 
    Q_publisher = layers.Reshape((-1,1))(Q_publisher)
    N = len(set(df['Year-Of-Publication']))
    Q_publication_year = Embedding(N, K, embeddings_regularizer=l2(reg))(publication_year)
    Q_publication_year = layers.Reshape((-1,1))(Q_publication_year)

    Q_concat = layers.Concatenate(axis=-1)([Q_item, Q_author, Q_publisher, Q_publication_year])
    Q_conv = layers.Conv1D(filters=1, kernel_size=1)(Q_concat)
    Q_embedding = Flatten()(Q_conv)

    # bias
    N = len(set(df['User-ID']))
    user_bias = Embedding(N, 1, embeddings_regularizer=l2(reg))(user)
    user_bias = Flatten()(user_bias)

    N = len(set(df['ISBN']))
    item_bias = Embedding(N, 1, embeddings_regularizer=l2(reg))(item)  
    item_bias = Flatten()(item_bias)
    R = Dot(axes=-1)([P_embedding, Q_embedding])                            
    R = Add()([R, user_bias, item_bias])
    R = layers.Activation('sigmoid')(R)
    model = Model([item, author, publisher, publication_year, user, location, age], R)
    return model

model = conv_mf_clf(ratings)
#model = mlp(ratings)
#model.summary()

model.compile(
  loss='binary_crossentropy',
  optimizer=Adam(lr=0.001),
  #optimizer=SGD(lr=0.07, momentum=0.9),
  metrics=['accuracy', f1_m],
)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,  patience=5)
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

result = model.fit(x=[X_train[feature].values for feature in features], y=(Y_train),
                   batch_size=100, epochs=100, 
                   validation_data=([X_test[feature].values for feature in features], (Y_test)),
                   callbacks=[reduce_lr, es],
                  class_weight={0: 0.2, 1: 0.8})


# In[3]:


#model.save('../model/clf1207.h5')
model.save('../model/clf1209.h5')


# In[ ]:




