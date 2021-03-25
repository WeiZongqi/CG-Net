import os
import math
import numpy as np
import json


def vector_product(coord):
    coord = np.array(coord).reshape((4,2))
    temp_det = 0
    for idx in range(3):
        temp = np.array([coord[idx],coord[idx+1]])
        temp_det +=np.linalg.det(temp)
    temp_det += np.linalg.det(np.array([coord[-1],coord[0]]))
    return temp_det*0.5


CLASSES = ('plane', 'baseball-diamond',
                'bridge', 'ground-track-field',
                'small-vehicle', 'large-vehicle',
                'ship', 'tennis-court',
                'basketball-court', 'storage-tank',
                'soccer-ball-field', 'roundabout',
                'harbor', 'swimming-pool',
                'helicopter')


root_path = '/home/gqx/GQX/AerialDetection/data/dota/train/labelTxt/'

txts = os.listdir(root_path)

txts_ = txts[:2]

dictt = {}
for c in CLASSES:
    dictt[c] = []

for txt in txts:
    path_txt = root_path + txt
    f = open(path_txt,'r')
    lines = f.readlines()
    lines = lines[2:]
    lines = [x.replace('\n','') for x in lines]
    for line in lines:
        l_s = line.split(' ')
        cls = l_s[8]
        coord = l_s[:8]
        coord = [int(x) for x in coord]
        ss = vector_product(coord)
        dictt[cls].append(int(ss))
    f.close()
# print(dictt)

f1 = open('size_about.json','w')
f1.write(json.dumps(dictt))
f1.close()