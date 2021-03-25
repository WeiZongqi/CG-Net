import os
import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPoint  # 多边形

# def overlap(list1, list2):
#     line1 = [2, 0, 2, 2, 0, 0, 0, 2]  # 四边形四个点坐标的一维数组表示，[x,y,x,y....]
#     a = np.array(line1).reshape(4, 2)  # 四边形二维坐标表示
#     poly1 = Polygon(a).convex_hull  # python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下 右下 右上 左上
#     print(Polygon(a).convex_hull)  # 可以打印看看是不是这样子
#
#     line2 = [1, 1, 4, 1, 4, 4, 1, 4]
#     b = np.array(line2).reshape(4, 2)
#     poly2 = Polygon(b).convex_hull
#     print(Polygon(b).convex_hull)
#
#     union_poly = np.concatenate((a, b))  # 合并两个box坐标，变为8*2
#     # print(union_poly)
#     print(MultiPoint(union_poly).convex_hull)  # 包含两四边形最小的多边形点
#     if not poly1.intersects(poly2):  # 如果两四边形不相交
#         iou = 0
#     else:
#         try:
#             inter_area = poly1.intersection(poly2).area  # 相交面积
#             print(inter_area)
#             # union_area = poly1.area + poly2.area - inter_area
#             union_area = MultiPoint(union_poly).convex_hull.area
#             print(union_area)
#             if union_area == 0:
#                 iou = 0
#             # iou = float(inter_area) / (union_area-inter_area) #错了
#             iou = float(inter_area) / union_area
#             # iou=float(inter_area) /(poly1.area+poly2.area-inter_area)
#             # 源码中给出了两种IOU计算方式，第一种计算的是: 交集部分/包含两个四边形最小多边形的面积
#             # 第二种： 交集 / 并集（常见矩形框IOU计算方式）
#         except shapely.geos.TopologicalError:
#             print('shapely.geos.TopologicalError occured, iou set to 0')
#             iou = 0
#     print(a)
#     print(iou)


baseline_path = 'D:/gqx/baseline'
ground_truth_path = 'D:/gqx/ground_truth'
transformer_path = 'D:/gqx/transformer'

baseline_txts = os.listdir(baseline_path)
ground_truth_txts = os.listdir(ground_truth_path)
transformer_txts = os.listdir(transformer_path)

# print(baseline_txts)
# print('======')
# print(ground_truth_txts[0])
# print('======')
# print(transformer_txts)
# print('======')

# for i in range(len(ground_truth_txts)):
#     ground_truth_file = ground_truth_path + '/' + ground_truth_txts[i]
#     transformer_file = transformer_path + '/' + transformer_txts[i]
#     baseline_file = baseline_path + '/' + baseline_txts[i]

BD_ground_truth_file = ground_truth_path + '/' + ground_truth_txts[0]
BD_transformer_file = transformer_path + '/' + transformer_txts[0]
BD_baseline_file = baseline_path + '/' + baseline_txts[0]

f1 = open(BD_ground_truth_file, 'r')
f2 = open(BD_transformer_file, 'r')
f3 = open(BD_baseline_file, 'r')

lines_ground_truth = f1.readlines()
lines_ground_truth = [x.replace('\n','') for x in lines_ground_truth]

lines_transformer = f2.readlines()
lines_transformer = [x.replace('\n','') for x in lines_transformer]

for line_ground_truth in lines_ground_truth:
    print(line_ground_truth)

# lines_baseline = f3.readlines()
# lines_baseline = [x.replace('\n','') for x in lines_baseline]

