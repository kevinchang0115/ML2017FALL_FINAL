import io
import os
import jieba
from gensim.models import word2vec
import numpy as np
import random
import json

jieba.dt.cache_file = 'jieba.cache.new'
jieba.set_dictionary('jieba_dict/dict.txt.big')

def parseData(data):
    return [(sent.replace('A:', '')).replace('B:', '') for sent in data]

def jiebaSeg(lines):
    segLine = []
    words = jieba.cut(lines)
    for word in words:
        if word != ' ' and word != '':
            segLine.append(word)
    return segLine

def make_test_data(data_file):
    index = []
    questions = []
    options = []
    answer = []
    with io.open(data_file, 'r', encoding = 'utf-8') as content:
        for line in content:
            lines = line.split(',')
            if len(lines) == 3:
                index.append(lines[0])
                dialog_ = lines[1].split('\t')
                questions.append(parseData(dialog_))
                options_ = lines[2].split('\t')
                options.append(parseData(options_))
                try:
                    answer.append(int(lines[3][:-1]))
                except:
                    pass
        del index[0]
        del questions[0]
        del options[0]

    quests = []
    opts = []
    for i, question in enumerate(questions):
        questSeg = []
        for q in question:
            qSeg = jiebaSeg(q)
            questSeg += [i for i in qSeg]
        
        for o in options[i]:
            oSeg = jiebaSeg(o)
            if oSeg is None:
                continue
            quests.append(questSeg)
            opts.append(oSeg)
        
    return quests, opts

	
train_file_1 = 'data/1_train.txt'
train_file_2 = 'data/2_train.txt'
train_file_3 = 'data/3_train.txt'
train_file_4 = 'data/4_train.txt'
train_file_5 = 'data/5_train.txt'
train_list = [train_file_1, train_file_2, train_file_3, train_file_4, train_file_5]

# Build segments and produce training data
output_file = 'data/trainSeg.txt'
output = io.open(output_file, 'w', encoding='utf-8')

quests = []
opts = []
for train_file in train_list:
    quest = []
    opt = []
    with io.open(os.path.join(train_file), 'r', encoding='utf-8') as content:
        for line in content:
            words = jieba.cut(line, cut_all=False)
            wordCount = 0
            segLine = []
            for id, word in enumerate(words):
                if word != '\n' and word != ' ' :
                    output.write(word + ' ')
                    wordCount = wordCount + 1
                if wordCount != 0:
                    output.write(u'\n')
                if word != ' ' and word != '':
                    segLine.append(word)
            quest.append(segLine)
            opt.append(segLine)
    quest.pop()
    opt.remove(opt[0])
    quests = quests + quest
    opts = opts + opt
output.close()
ans = np.ones(len(quests))

# Produce wrong options
quests_wrong = []
opts_wrong = []
for line in quests:
    repeat = 1
    for i in range(repeat):
        quests_wrong.append(line)
        opts_wrong.append(opts[random.randint(0,len(opts)-1)])
ans_wrong = np.zeros(len(quests_wrong))
quests = quests + quests_wrong
opts = opts + opts_wrong
ans = np.concatenate((ans, ans_wrong), axis=0)

# Build word vector
dim = 50
sentences = word2vec.Text8Corpus('data/trainSeg.txt')
model = word2vec.Word2Vec(sentences, size=dim, min_count = 1)
weights = np.concatenate((np.zeros((1,dim)), model.wv.syn0), axis=0)
dic = dict([(k, v.index+1) for k, v in model.wv.vocab.items()])

if not os.path.exists('model'):
    os.mkdir('model')
model.save("model/final_tradition.w2v")
model.wv.save_word2vec_format("model/word2vecFormat_tradition", binary=False)
np.save("model/weights.npy", weights)
with open("model/dict", 'w') as f:
    f.write(json.dumps(dic))
with open("data/training_quests", 'w') as f:
    f.write(json.dumps(quests))
with open("data/training_opts", 'w') as f:
    f.write(json.dumps(opts))
np.save("data/ans.npy", ans)

s = np.zeros(len(quests))
for i in range(len(quests)):
    s[i] = len(quests[i])
print('training question length: %f' % (np.mean(s) + np.std(s)))

s = np.zeros(len(opts))
for i in range(len(opts)):
    s[i] = len(opts[i])
print('training option length: %f' % (np.mean(s) + np.std(s)))

test_data = "data/testing_data.csv"
quests, opts = make_test_data(test_data)
with open("data/testing_quests", 'w') as f:
    f.write(json.dumps(quests))
with open("data/testing_opts", 'w') as f:
    f.write(json.dumps(opts))

s = np.zeros(len(quests))
for i in range(len(quests)):
    s[i] = len(quests[i])
print('testing question length: %f' % (np.mean(s) + np.std(s)))

s = np.zeros(len(opts))
for i in range(len(opts)):
    s[i] = len(opts[i])
print('testing option length: %f' % (np.mean(s) + np.std(s)))