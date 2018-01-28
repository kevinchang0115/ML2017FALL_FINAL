# ML2017 FALL - Final Project
Topic: conversations in TV shows

## Introduction
It's the final project of 2017 Machine Learning (Fall) course of NTU.

We use collected Chinese conversations in TV shows as training data to train ML models and answer questions relevant to the TV shows.

### Data sources
The training data, "1_train.txt" - "5_train.txt" are provided by TAs and testing data (questions and options) are from classmates.

### Rules
see details on: https://docs.google.com/presentation/d/1XBuGnr-QO0CRoxUaN1d7i8iwDu_3WTTtIOJW8Q2N1EU/edit#slide=id.g29b0e46830_0_0

### Packages
python 3.5

numpy 1.12

pandas 0.20

jieba-zh_TW (uploaded)

gensim 3.0.1

tensorflow 1.2.1

keras 2.0.8

## Models
We use 3 kinds of models to make predictions and do the plurarity voting for best answers.
(1) Similarity model (using similarity of sentences between the question and options)
(2) LSTM-DNN model
(3) LSTM-Dot model

## Run
### Using uploaded models ("best_dnn.h5" and "best_dot.h5" in /model)
run:
```
bash final.sh
```
/result folder will be created, and four predicted files will be saved in this folder.

The 4 predicted files are "ans_sim.csv" (similarity model), "ans_best_dnn.csv" (lstm-dnn model, saved in /model), "ans_best_dot.csv" 

(lstm-dot model, saved in /model), "vote.csv" (voted results of above 3 models).

### Using new training models
run:
```
bash final_train.sh
```
"train.py" will be executed, and new training models, "final_dnn.h5" and "final_dot.h5" will be saved in /model.

The predicted files will be produced according to new training models instead of uploaded models.
