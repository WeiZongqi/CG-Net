import os
import json

json_path = '/home/wzq/remote2/AerialDetection/tools/analysis/train_ground.json'

with open(json_path, 'r') as load_json:
    json_data = json.load(load_json)

    # print(json_data)
    # print(json_data.keys())
    # print(json_data['plane'][0])


root_path = '/home/wzq/remote2/AerialDetection/work_dirs/faster_rcnn_RoITrans_r50_fpn_selfatten6_1_1x_dota_test/ground_truth/'

for cls in json_data.keys():
    file = root_path + cls + '.txt'
    f = open(file, 'w')
    cls_lines = [x + '\n' for x in json_data[cls]]
    f.writelines(cls_lines)
    f.close()




# for i in range(len(json_data['annotations'])):
#     category_id = json_data['annotations'][i]['category_id']
#     res[category_id - 1].append(json_data['annotations'][i])

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