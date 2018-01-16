import os
import sys
import numpy as np
import pandas as pd

def vote(ans):
    ans = ans.astype(int)
    option = np.zeros(max(ans)+1)
    for a in ans:
        option[a] += 1
    M = np.where(option == max(option))[0]
    if len(M) == 1:
        return M[0]
    else:
        for a in ans:
            for m in M:
                if a == m:
                    return a

file1 = os.path.join('result', sys.argv[1])
file2 = os.path.join('result', sys.argv[2])
file3 = os.path.join('result', sys.argv[3])
file_list = [file1, file2, file3]
ans = np.zeros((5060, len(file_list)))
for i, file in enumerate(file_list):
    ans[:,i] = pd.read_csv(file)['ans']
    
A = np.zeros(len(ans))
for i in range(len(ans)):
    A[i] = vote(ans[i])

output_dir = 'result'
file_name = 'vote.csv'
output_path = os.path.join(output_dir, file_name)
with open(output_path, 'w') as f:
    f.write('id,ans\n')
    for i in range(len(A)):    
        f.write('%d,%d\n' %(i+1, A[i]))
