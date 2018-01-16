#!/bin/bash

python3 build.py
python3 sim.py
python3 test.py best_dot.h5
python3 test.py best_dnn.h5
python3 vote.py ans_sim.csv ans_best_dot.csv ans_best_dnn.csv
