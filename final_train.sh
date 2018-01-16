#!/bin/bash

python3 build.py
python3 sim.py
python3 train.py
python3 test.py final_dot.h5
python3 test.py final_dnn.h5
python3 vote.py ans_sim.csv ans_final_dot.csv ans_final_dnn.csv
