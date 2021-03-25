import os
import cv2 as cv


root = '/home/gqx/GQX/AerialDetection/data/dota/'

labeltxts_train = os.listdir(root + 'train/labelTxt/')
labeltxts_val = os.listdir(root + 'val/labelTxt/')

p1 = '/home/wzq/datasets/dota/dota_select/images/'
p2 = '/home/wzq/datasets/dota/dota_select/labelTxt/'

if not os.path.exists(p1):
    os.makedirs(p1)
if not os.path.exists(p2):
    os.makedirs(p2)

for txt in labeltxts_train:
    f = open(root + 'train/labelTxt/' + txt, 'r')
    lines = f.readlines()
    f.close()

    lines_ = [x.split('\n')[0] for x in lines]
    lines_ = lines_[2:]
    for line in lines_:
        # print(line,'===')
        if line.split(' ')[8] == 'baseball-diamond':
            f1 = open(p2 + txt, 'w')
            f1.writelines(lines)
            f1.close()
            img = cv.imread(root + 'train/images/'+txt.replace('txt','png'))
            cv.imwrite(p1 + txt.replace('txt','png'), img)
            break

for txt in labeltxts_val:
    f = open(root + 'val/labelTxt/' + txt, 'r')
    lines = f.readlines()
    f.close()

    lines_ = [x.split('\n')[0] for x in lines]
    lines_ = lines_[2:]
    for line in lines_:
        if line.split(' ')[8] == 'baseball-diamond':
            f1 = open(p2 + txt, 'w')
            f1.writelines(lines)
            f1.close()
            img = cv.imread(root + 'val/images/' + txt.replace('txt', 'png'))
            cv.imwrite(p1 + txt.replace('txt', 'png'), img)
            break