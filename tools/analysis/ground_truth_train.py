import os
import json

json_path = '/home/gqx/GQX/AerialDetection/data/dota1-split-1024/trainval1024/DOTA_trainval1024.json'

with open(json_path, 'r') as load_json:
    json_data = json.load(load_json)
    #print(json_data)
    # print(json_data.keys())
    # print(json_data['annotations'])
    # print(type(json_data['annotations']))
    # print('========')
    # print(json_data['annotations'][0])
    # print(json_data['annotations'][1])
    # print(json_data['annotations'][2])
    # print('========')
    # print(json_data['annotations'][0].keys())
    # print('========')
    # print(len(json_data['annotations']))
    # print('========')
    # print(json_data.keys())
    print(json_data['categories'])

res = []
for i in range(15):
    res.append([])


for i in range(len(json_data['annotations'])):
    category_id = json_data['annotations'][i]['category_id']
    res[category_id - 1].append(json_data['annotations'][i])

# print(res[0])
# print('======')
# print(res[1])

# print(res[0][0])
# image_id = res[0][0]['image_id']
# bbox = res[0][0]['bbox']
# print('======')
# print(type(image_id))
# print(bbox)

# for j in range(len(res)):
#     root_path = '/home/wzq/remote2/AerialDetection/work_dirs/faster_rcnn_RoITrans_r50_fpn_selfatten6_1_1x_dota_test/ground_truth/'
#     file = root_path + str(j+1) + '.txt'
#     # print(file)
#     f = open(file, 'w')
#     for k in range(len(res[j])):
#         image_id = res[j][k]['image_id']
#         bbox = res[j][k]['bbox']
#         img = 'P' + str(image_id)
#         f.write(img + ' ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' + str(bbox[3]) + ' ' + '\n')
#     f.close()