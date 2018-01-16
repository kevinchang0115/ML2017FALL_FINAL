import os
import sys
import json
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import sequence
from keras.utils.generic_utils import get_custom_objects

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean((1-y_true) * 0.5 * K.square(y_pred) + 
                  0.5 * y_true * K.square(K.maximum(margin - y_pred, 0)))
    
get_custom_objects().update({"contrastive_loss": contrastive_loss})

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

with open("model/dict", 'r') as f:
    dic = json.loads(f.read())
with open("data/testing_quests", 'r') as f:
    quests = json.loads(f.read())
with open("data/testing_opts", 'r') as f:
    opts = json.loads(f.read())    

quests = word2idx(quests, dic)
opts = word2idx(opts, dic)

maxlen_q = 20
maxlen_o = 10
quests = sequence.pad_sequences(quests, maxlen=maxlen_q)
opts = sequence.pad_sequences(opts, maxlen=maxlen_o)

# predict with LSTM-DNN model
model_name = sys.argv[1]
model_path = os.path.join('model', model_name+'_dnn.h5')
model = load_model(model_path)
model.summary()

score = model.predict([quests, opts])
score = score.reshape((len(score)//6,6))

output_dir = 'result'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
file_name = 'ans_'+ model_name + '_dnn.csv'
output_path = os.path.join(output_dir, file_name)
with open(output_path, 'w') as f:
    f.write('id,ans\n')
    for i in range(len(score)):
        highest = max(score[i])
        ans = np.where(score[i] == highest)[0][0]            
        f.write('%d,%d\n' %(i+1, ans))

# predict with LSTM-Dot model
model_path = os.path.join('model', model_name+'_dot.h5')
model = load_model(model_path)
model.summary()

score = model.predict([quests, opts])
score = score.reshape((len(score)//6,6))

file_name = 'ans_'+ model_name + '_dot.csv'
output_path = os.path.join(output_dir, file_name)
with open(output_path, 'w') as f:
    f.write('id,ans\n')
    for i in range(len(score)):
        highest = max(score[i])
        ans = np.where(score[i] == highest)[0][0]            
        f.write('%d,%d\n' %(i+1, ans))
