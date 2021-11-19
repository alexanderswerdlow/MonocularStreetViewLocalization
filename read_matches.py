import cv2
import csv

from collections import defaultdict
from config import openmvg_data

matches = defaultdict(list)
keypoints = defaultdict(list)
with open(f'{openmvg_data}/matches.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        n1,x1,y1,n2,x2,y2 = row
        k1 = cv2.KeyPoint(float(x1), float(y1), 1)
        k2 = cv2.KeyPoint(float(x2), float(y2), 1)
        
        d = cv2.DMatch(len(keypoints[n1]) + 1, len(keypoints[n2]) + 1, 1)
        k = sorted([n1, n2])
        matches[k[0] + ' ' + k[1]].append([d])

        keypoints[n1].append(k1)
        keypoints[n2].append(k2)

for k, desc in matches.items():
    if 'frame' in k:
        n1, n2 = k.split(' ')
        reference_img = cv2.drawMatchesKnn(cv2.imread(f'{openmvg_data}/{n1}'), keypoints[n1], cv2.imread(f'{openmvg_data}/{n2}'), keypoints[n2], desc, None, flags=2)
        cv2.imwrite(f'tmp/{n1}+{n2}', reference_img)