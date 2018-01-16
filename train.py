import numpy as np
from keras.models import Model
from keras.layers import Embedding, Dense, Dropout, LSTM, Flatten
from keras.layers import Input, Concatenate, Dot
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import sequence
import json
from keras import backend as K

def word2idx(x_data, dic):
    x_data_idx = []
    for i in range(len(x_data)):
        x_data_idx.append([])
        for j in range(len(x_data[i])):        
            if dic.get(x_data[i][j]) != None:
                x_data_idx[i].append(dic[x_data[i][j]])
            else:
                x_data_idx[i].append(0)
    x_data_idx = np.array(x_data_idx)
    return x_data_idx

def shuffle(x_users, x_movies, y_rating):
    rand = np.arange(len(y_rating))
    np.random.shuffle(rand)
    return (x_users[rand], x_movies[rand], y_rating[rand])

weights = np.load("model/weights.npy")
with open("model/dict", 'r') as f:
    dic = json.loads(f.read())
with open("data/training_quests", 'r') as f:
    quests = json.loads(f.read())
with open("data/training_opts", 'r') as f:
    opts = json.loads(f.read())
ans = np.load("data/ans.npy")

quests = word2idx(quests, dic)
opts = word2idx(opts, dic)

maxlen_q = 20
maxlen_o = 10
quests = sequence.pad_sequences(quests, maxlen=maxlen_q)
opts = sequence.pad_sequences(opts, maxlen=maxlen_o)

def lstm_dot(maxlen_q, maxlen_o, weights, rate):
    input_quest = Input(shape=(maxlen_q,))
    input_opt = Input(shape=(maxlen_o,))
    embedding = Embedding(weights.shape[0], weights.shape[1], weights=[weights])
    embed_quest = embedding(input_quest)
    embed_opt = embedding(input_opt)
    lstm_quest = LSTM(weights.shape[1])(embed_quest)
    lstm_quest = Dropout(rate)(lstm_quest)
    lstm_opt = LSTM(weights.shape[1])(embed_opt)
    lstm_opt = Dropout(rate)(lstm_opt)
    out = Dot(axes=1, normalize=True)([lstm_quest, lstm_opt])    
    return Model([input_quest, input_opt], out)

def lstm_dnn(maxlen_q, maxlen_o, weights, rate):
    input_quest = Input(shape=(maxlen_q,))
    input_opt = Input(shape=(maxlen_o,))
    embedding = Embedding(weights.shape[0], weights.shape[1], weights=[weights])
    embed_quest = embedding(input_quest)
    embed_opt = embedding(input_opt)
    lstm_quest = LSTM(weights.shape[1])(embed_quest)
    lstm_quest = Dropout(rate)(lstm_quest)
    lstm_opt = LSTM(weights.shape[1])(embed_opt)
    lstm_opt = Dropout(rate)(lstm_opt)
    concat = Concatenate(axis=1)([lstm_quest, lstm_opt])
    dnn = Dense(weights.shape[1], activation='relu')(concat)
    dnn = Dropout(rate)(dnn)
    out = Dense(1, activation='sigmoid')(dnn)    
    return Model([input_quest, input_opt], out)
    
def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean((1-y_true) * 0.5 * K.square(y_pred) + 
                  0.5 * y_true * K.square(K.maximum(margin - y_pred, 0)))

rate = 0
#lstmModel = lstm_dnn(maxlen_q, maxlen_o, weights, rate)
lstmModel = lstm_dot(maxlen_q, maxlen_o, weights, rate)
lstmModel.compile(loss=contrastive_loss, optimizer='adam')
lstmModel.summary()

quests, opts, ans = shuffle(quests, opts, ans)

batch_size = 16383
epochs = 100

earlystop = EarlyStopping(monitor='val_loss', patience=4)
checkpoint = ModelCheckpoint("model/lstm_dot.h5", 
                             monitor='val_loss', save_best_only=True)
callbacks=[earlystop, checkpoint]

hist = lstmModel.fit([quests, opts], ans, 
                 batch_size = batch_size, epochs = epochs, 
                 validation_split = 0.1, callbacks=callbacks)

lstmModel = lstm_dnn(maxlen_q, maxlen_o, weights, rate)
lstmModel.compile(loss=contrastive_loss, optimizer='adam')
lstmModel.summary()

quests, opts, ans = shuffle(quests, opts, ans)

batch_size = 16383
epochs = 100

earlystop = EarlyStopping(monitor='val_loss', patience=4)
checkpoint = ModelCheckpoint("model/lstm_dnn.h5", 
                             monitor='val_loss', save_best_only=True)
callbacks=[earlystop, checkpoint]

hist = lstmModel.fit([quests, opts], ans, 
                 batch_size = batch_size, epochs = epochs, 
                 validation_split = 0.11, callbacks=callbacks)