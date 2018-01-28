# ML2017 FALL - Final Project
Topic: TV conversation

## Introduction
Rules: https://docs.google.com/presentation/d/1XBuGnr-QO0CRoxUaN1d7i8iwDu_3WTTtIOJW8Q2N1EU/edit#slide=id.g29b0e46830_0_0

## Packages
python 3.5

numpy 1.12

pandas 0.20

jieba-zh_TW (uploaded)

gensim 3.0.1

tensorflow 1.2.1

keras 2.0.8

## Run
### Using uploaded models
run:
```
bash final.sh
```
/result folder will be created, and four predicted files will be saved in this folder.
The 4 predicted files are "ans_sim.csv" (similarity model), "ans_dnn.csv" (lstm & dnn model, saved in /model), "ans_dot.csv" (lstm & dot model, saved in /model), "vote.csv" (voted results of above 3 models).

### Using new training models
```
bash final_train.sh
```
"train.py" will be executed, and new training models, "final_dnn.h5" and "final_dot.h5" will be saved in /model.
The predicted files will be produced according to new training models instead of uploaded models.

### Example
```
bash final.sh
```
,and find output file in /result/vote.csv
