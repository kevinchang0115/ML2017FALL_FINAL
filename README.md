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
### Using uploaded model (in /model dir)
```
bash final.sh $1
```
$1: file name of predicted results (saved in /result)

(/result will be created, and all predicted files including the voted one will be saved in this folder.)

### Using new training model
'''
bash final_train.sh $1
'''
$1: file name of predicted results (saved in /result)

### Example
```
bash final.sh pred.csv
```
,and find output file in /src/result/pred.csv
