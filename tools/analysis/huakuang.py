import os
import cv2 as cv

out_path = '/home/wzq/datasets/dota/dota_select/test_out/'

# img_name = 'P0117_out.jpg'

if not os.path.exists(out_path.replace('test_out','test_out_gt')):
    os.mkdir(out_path.replace('test_out','test_out_gt'))

imgs_name = os.listdir(out_path)
for img_name in imgs_name:

    save_path = out_path.replace('test_out','test_out_gt') + img_name

    img = cv.imread(out_path + img_name)
    txt_path = out_path.replace('test_out','labelTxt') + img_name.replace('_out','').replace('jpg','txt')
    f = open(txt_path,'r')
    lines = f.readlines()
    f.close()

    lines = lines[2:]
    for line in lines:
        line = line.split(' ')[:8]
        line = [int(x) for x in line]
        cv.line(img, (line[0],line[1]),(line[2],line[3]),(0, 0, 255), 2)
        cv.line(img, (line[4],line[5]),(line[2],line[3]),(0, 0, 255), 2)
        cv.line(img, (line[4],line[5]),(line[6],line[7]),(0, 0, 255), 2)
        cv.line(img, (line[0],line[1]),(line[6],line[7]),(0, 0, 255), 2)

    cv.imwrite(save_path,img)


# cv.imshow('test',img)
# cv.waitKey()