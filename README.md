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
```
cd src
```
In /src folder, run:
```
bash final.sh $1 $2
```
$1: file path of testing data

$2: file name of final predicted results (saved in /result)

(/result will be created, and all predicted files including the final one will be saved in this folder.)

### Example
```
bash final.sh data/testing_data.csv pred.csv
```
,and find output file in /src/result/pred.csv
