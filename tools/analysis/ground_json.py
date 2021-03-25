import os
import math
import numpy as np
import json


def getTxtInfo(root_path):
    txts = os.listdir(root_path)

    for txt in txts:
        image_id = txt.split('.')[0]

        path_txt = root_path + txt
        f = open(path_txt, 'r')
        lines = f.readlines()
        lines = lines[2:]
        lines = [x.replace('\n', '') for x in lines]
        for line in lines:
            l_s = line.split(' ')
            cls = l_s[8]
            # coord = l_s[:8]
            coord = line.split(' ' + cls)[0]
            dictt[cls].append(image_id + ' ' + coord)
            # ss = vector_product(coord)
            # dictt[cls].append(int(ss))
        f.close()
    # print(dictt)



CLASSES = ('plane', 'baseball-diamond',
                'bridge', 'ground-track-field',
                'small-vehicle', 'large-vehicle',
                'ship', 'tennis-court',
                'basketball-court', 'storage-tank',
                'soccer-ball-field', 'roundabout',
                'harbor', 'swimming-pool',
                'helicopter')


root_path1 = '/home/gqx/GQX/AerialDetection/data/dota/train/labelTxt/'
root_path2 = '/home/gqx/GQX/AerialDetection/data/dota/val/labelTxt/'


dictt = {}
for c in CLASSES:
    dictt[c] = []

getTxtInfo(root_path1)
getTxtInfo(root_path2)

f1 = open('train_ground.json', 'w')
f1.write(json.dumps(dictt))
f1.close()
