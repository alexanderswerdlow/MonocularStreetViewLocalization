import cv2
import csv

from collections import defaultdict

matches = {}
with open('/Volumes/GoogleDrive/Shared drives/EE209AS/data/matches_first_frame.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        n1,x1,y1,n2,x2,y2 = row
        k1 = cv2.KeyPoint(float(x1), float(y1), 1)
        k2 = cv2.KeyPoint(float(x2), float(y2), 1)
        
        if matches.get(n2) is None:
            matches[n2] = [[], [], []]
        
        d = cv2.DMatch(len(matches[n2][0]) + 1, len(matches[n2][0]) + 1, 1)
        matches[n2][0].append(k1)
        matches[n2][1].append(k2)
        matches[n2][2].append([d])

for ex, val in matches.items():
    print(ex)
    reference_img = cv2.drawMatchesKnn(cv2.imread(f'tmp/0-frame.jpg'), val[0], cv2.imread(f'tmp/{ex}'), val[1], val[2], None, flags=2)
    cv2.imwrite(f'tmp/1-{ex}', reference_img)