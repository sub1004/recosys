#!/usr/bin/env python
# coding: utf-8

# In[1]:

import keras
import keras.backend as KB
import pandas as pd
from keras import layers
from keras.models import Model
from keras.layers import Input, Embedding, Dot, Add, Flatten, Dropout, Dense
from keras.regularizers import l2
from keras.optimizers import SGD, Adam, rmsprop
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

def conv_mf(df, K=10, reg=0.0005):
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

    ### MF
    R = Dot(axes=-1)([P_embedding, Q_embedding])                            
    R = Add()([R, user_bias, item_bias])

    model = Model([item, author, publisher, publication_year, user, location, age], R)
    return model

def factorization_machine(f_size, k_latent=5, embedding_reg=0.0005):
    def get_embed(x_input, x_size, k_latent):
        if x_size > 0: #category
            embed = Embedding(x_size, k_latent, embeddings_regularizer=l2(embedding_reg))(x_input)
            embed = Flatten()(embed)
        else:
            embed = Dense(k_latent, kernel_regularizer=l2(embedding_reg))(x_input)
            #embed = Dense(k_latent)(x_input)
        return embed

    dim_input = len(f_size)
    input_x = [Input(shape=(1,)) for i in range(dim_input)] 
    biases = [get_embed(x, size, 1) for (x, size) in zip(input_x, f_size)]
    factors = [get_embed(x, size, k_latent) for (x, size) in zip(input_x, f_size)]
    s = Add()(factors)
    diffs = [layers.Subtract()([s, x]) for x in factors]
    dots = [layers.Dot(axes=1)([d, x]) for d,x in zip(diffs, factors)]
    dots = Add()(dots)
    dots_sum = layers.Lambda(lambda x: x/2)(dots)
    biases_sum = Add()(biases)
    x = Add()([dots_sum, biases_sum])
    model = Model(inputs=input_x, outputs=x)
    #output_f = factors + biases
    #model_features = Model(inputs=input_x, outputs=output_f)
    #model, model_features = build_model_1(X_train, f_size)
    return model
def svdpp(ratings, Ru_train, Ru_test, K=10, reg=0.00001):
    #svd++
    
    N = len(set(ratings['User-ID']))       # Number of users
    M = len(set(ratings['ISBN']))       # Number of users
    # Keras model
    user = Input(shape=(1,))            # User input
    item = Input(shape=(1,))            # Item input
    Y = Input(shape=(M,))
    random_normal = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
    P_embedding = Embedding(N, K, embeddings_regularizer=l2(reg), embeddings_initializer=random_normal)(user)     # (N, 1, K)
    Y_embedding = layers.Dense(K, kernel_regularizer=l2(reg), kernel_initializer=random_normal)(Y)
    P_embedding = Add()([P_embedding, Y_embedding])
    Q_embedding = Embedding(M, K, embeddings_regularizer=l2(reg))(item)     # (M, 1, K)
    
    user_bias = Embedding(N, 1, embeddings_regularizer=l2(reg), embeddings_initializer=keras.initializers.Zeros())(user)       # User bias term (N, 1, 1)
    item_bias = Embedding(M, 1, embeddings_regularizer=l2(reg), embeddings_initializer=keras.initializers.Zeros())(item)       # Item bias term (M, 1, 1)
    
    R = Dot(axes=2)([P_embedding, Q_embedding])                             # (N, 1, 1)
    R = Add()([R, user_bias, item_bias])
    R = Flatten()(R)                                                            
    # Model setting
    model = Model(inputs=[item, user, Y], outputs=R)
    return model

