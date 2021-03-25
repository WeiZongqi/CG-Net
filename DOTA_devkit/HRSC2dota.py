import math
import xml.etree.cElementTree as ET
from xml.dom.minidom import Document
import xml.dom.minidom
import numpy as np
import os
import cv2 as cv




def coordinate_convert_r(box):
    w, h = box[2:-1]
    theta = -box[-1]
    x_lu, y_lu = -w/2, h/2
    x_ru, y_ru = w/2, h/2
    x_ld, y_ld = -w/2, -h/2
    x_rd, y_rd = w/2, -h/2

    x_lu_ = math.cos(theta)*x_lu + math.sin(theta)*y_lu + box[0]
    y_lu_ = -math.sin(theta)*x_lu + math.cos(theta)*y_lu + box[1]

    x_ru_ = math.cos(theta) * x_ru + math.sin(theta) * y_ru + box[0]
    y_ru_ = -math.sin(theta) * x_ru + math.cos(theta) * y_ru + box[1]

    x_ld_ = math.cos(theta) * x_ld + math.sin(theta) * y_ld + box[0]
    y_ld_ = -math.sin(theta) * x_ld + math.cos(theta) * y_ld + box[1]

    x_rd_ = math.cos(theta) * x_rd + math.sin(theta) * y_rd + box[0]
    y_rd_ = -math.sin(theta) * x_rd + math.cos(theta) * y_rd + box[1]

    convert_box = [x_lu_, y_lu_, x_ru_, y_ru_, x_rd_, y_rd_, x_ld_, y_ld_]

    return convert_box

def change(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    box_list = []
    dictt = {}
    for child1 in root:
        if child1.tag == 'Img_ID':
            dictt['img_id'] = child1.text
        if child1.tag == 'Img_SizeWidth':
            img_width = int(child1.text)
            dictt['width'] = img_width
        if child1.tag == 'Img_SizeHeight':
            img_height = int(child1.text)
            dictt['height'] = img_height
        if child1.tag == 'HRSC_Objects':

            for child2 in child1:
                if child2.tag == 'HRSC_Object':
                    label = 1
                    tmp_box = [0., 0., 0., 0., 0.]
                    difficulty = 0
                    for node in child2:
                        if node.tag == 'mbox_cx':
                            tmp_box[0] = float(node.text)
                        if node.tag == 'mbox_cy':
                            tmp_box[1] = float(node.text)
                        if node.tag == 'mbox_w':
                            tmp_box[2] = float(node.text)
                        if node.tag == 'mbox_h':
                            tmp_box[3] = float(node.text)
                        if node.tag == 'mbox_ang':
                            tmp_box[4] = float(node.text)
                        if node.tag == 'difficult':
                            difficulty = int(node.text)

                    tmp_box = coordinate_convert_r(tmp_box)
                    # tmp_box.append('ship')
                    tmp_box.append(label)
                    tmp_box.append(difficulty)
                    box_list.append(tmp_box)
    gtbox_label = np.array(box_list, dtype=np.int32)
    dictt['box'] = gtbox_label
    # dictt['box'] = box_list
    # print(dictt)
    return dictt

if __name__ == '__main__':
    # xml_path = '/home/gqx/GQX/AerialDetection/data/HRSC2016/Train/Annotations/100000638.xml'
    root_path = '/home/gqx/GQX/AerialDetection/data/HRSC2016/'
    save_path = '/home/wzq/datasets/HRSC/'

    train_imgs = os.listdir(root_path+'train/images/')
    for img_ in train_imgs:
        img_path = root_path+'train/images/' + img_
        img = cv.imread(img_path)
        if img.all() == None:
            print(img_)
        cv.imwrite(save_path + 'train/images/'+ img_.replace('bmp','png'),img)

    test_imgs = os.listdir(root_path + 'test/images/')
    for img_ in test_imgs:
        img_path = root_path + 'test/images/' + img_
        img = cv.imread(img_path)
        cv.imwrite(save_path + 'test/images/' + img_.replace('bmp', 'png'), img)





    train_xml = os.listdir(root_path+'Train/Annotations/')
    for p1 in train_xml:
        xml_path = root_path+'Train/Annotations/' + p1
        dictt = change(xml_path)
        # print(dictt)
        fo = open(save_path+'train/labelTxt/'+p1.replace('xml','txt'),'w')
        fo.write('imagesource:GoogleEarch\n')
        fo.write('gsd:null\n')
        for box in dictt['box']:
            print(box)
            box_ = str(box[0]) + ' ' + str(box[1]) + ' ' +str(box[2]) + ' ' +str(box[3]) + ' ' + \
                    str(box[4]) + ' ' +str(box[5]) + ' ' +str(box[6]) + ' ' +str(box[7]) + ' ' \
                   + 'ship ' + str(box[9])
            fo.write(box_)
            fo.write('\n')
        fo.close()

    test_xml = os.listdir(root_path + 'Test/Annotations/')
    for p1 in test_xml:
        xml_path = root_path + 'Test/Annotations/' + p1
        dictt = change(xml_path)
        # print(dictt)
        fo = open(save_path + 'test/labelTxt/' + p1.replace('xml', 'txt'), 'w')
        fo.write('imagesource:GoogleEarch\n')
        fo.write('gsd:null\n')
        for box in dictt['box']:
            print(box)
            box_ = str(box[0]) + ' ' + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' ' + \
                   str(box[4]) + ' ' + str(box[5]) + ' ' + str(box[6]) + ' ' + str(box[7]) + ' ' \
                   + 'ship ' + str(box[9])
            fo.write(box_)
            fo.write('\n')
        fo.close()






