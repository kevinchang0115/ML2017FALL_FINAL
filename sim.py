import io
import os
import jieba
import numpy as np
from gensim.models import word2vec

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

def sim(quest, opt):
    max_sim = 0.
    max_sim2 = 0.
    opt_len = len(opt)
    
    questSeg = []
    optSeg = []
    
    for q in quest:
        qSeg = jiebaSeg(q)
        questSeg += [i for i in qSeg]
    
    for o in opt:
        oSeg = jiebaSeg(o)
        if oSeg is None:
            continue
        optSeg.append(oSeg)
        
    sum_sim = np.zeros((opt_len, 1), dtype=float)
    threshold = 0.11
    for i, one_opt in enumerate(optSeg):
        count_sim_time = 0.
        for k, dialog_seg in enumerate(questSeg):
            for opt_seg in one_opt:
                try:
                    sim = abs(model.similarity(dialog_seg, opt_seg))
                    if sim > threshold:
                        sum_sim[i] = sum_sim[i] + sim
                        count_sim_time = count_sim_time + 1
                except Exception as e:
                    pass
    return sum_sim

def checkSim(data_file):
    index = []
    questions = []
    options = []
    answer = []
    model_output = []
    model_output2 = []
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
    
    M = []
    for i, question in enumerate(questions):
        mat = sim(question, options[i])
        M.append(mat)
    return M


# Load word2vector
model = word2vec.Word2Vec.load("model/word2vec.w2v")

# Calculate similarity
test_data = "data/testing_data.csv"
M = checkSim(test_data)

# Write results
output_dir = 'result'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
file_name = 'ans_sim.csv'
output_path = os.path.join(output_dir, file_name)
with open(output_path, 'w') as f:
    f.write('id,ans\n')
    for i in range(len(M)):
        ans_sim = max(M[i])
        ans = np.where(M[i] == ans_sim)[0][0]            
        f.write('%d,%d\n' %(i+1, ans))
