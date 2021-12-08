import cv2
import os
import subprocess
from config import openmvg_data

OPENMVG_SFM_BIN = "/home/aswerdlow/github/openmvg_build/Linux-x86_64-RELEASE"

def get_kvld_matches_fast(cur_frame, panos):
    frame_id, frame = cur_frame
    frame_file = os.path.abspath(f'tmp_data/{frame_id}.jpg')
    cv2.imwrite(frame_file, frame)
    processes = []
    for pano_id, (pano, pano_frame) in panos.items():
        pano_file = os.path.abspath(f'tmp_data/{pano_id}.jpg')
        cv2.imwrite(pano_file, pano_frame)
        with open(os.devnull, 'w') as devnull: 
            processes.append(subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_sample_features_kvld"), "--img1", frame_file, "--img2", pano_file, "-o" , os.path.abspath(f'tmp_data'), '-d', '0', '-f', 'default'], stderr=devnull, stdout=subprocess.PIPE, encoding='utf8'))
    
    all_matches = []
    for p in processes:
        p.wait()
        matches = list(map(lambda a: a[2:].rstrip().split(','), filter(lambda x: x.startswith("m:"), p.stdout.readlines())))
        k1, k2, desc = [cv2.KeyPoint(float(m[0]), float(m[1]), 1) for m in matches], [cv2.KeyPoint(float(m[2]), float(m[3]), 1) for m in matches], [[cv2.DMatch(i + 1, i + 1, 1)] for i in range(len(matches))]
        reference_img = cv2.drawMatchesKnn(frame, k1, pano_frame, k2, desc, None, flags=2)
        cv2.imwrite(f'tmp-kvld/{frame_id}+{pano_id}.jpg', reference_img)
        all_matches.append([(pano, pano_frame), [p.pt for p in k1], [p.pt for p in k2], desc])
    return all_matches


def get_kvld_matches(cur_frame, panos, start_frame):
    frame_id, frame = cur_frame
    frame_file = os.path.abspath(f'working/{start_frame}/{frame_id}.jpg')
    cv2.imwrite(frame_file, frame)
    all_matches = []
    for pano_id, p in panos.items():
        pano_file = os.path.abspath(f'working/{start_frame}/{pano_id}.jpg')
        cv2.imwrite(pano_file, p[1])
        pKvld = subprocess.run( [os.path.join(OPENMVG_SFM_BIN, "openMVG_sample_features_kvld"), "--img1", frame_file, "--img2", pano_file], stdout=subprocess.PIPE, text=True, stderr=subprocess.DEVNULL).stdout.splitlines()
        matches = list(map(lambda a: a[2:].rstrip().split(','), filter(lambda x: x.startswith("m:"), pKvld)))
        print(pKvld[0], f'{len(matches)} matches for {frame_id}+{pano_id}')
        k1, k2, desc = [cv2.KeyPoint(float(m[0]), float(m[1]), 1) for m in matches], [cv2.KeyPoint(float(m[2]), float(m[3]), 1) for m in matches], [[cv2.DMatch(i + 1, i + 1, 1)] for i in range(len(matches))]
        # reference_img = cv2.drawMatchesKnn(frame, k1, p[1], k2, desc, None, flags=2)
        # cv2.imwrite(f'tmp_save/{frame_id}+{pano_id}.jpg', reference_img)
        all_matches.append([(p[0], p[2]), [p.pt for p in k1], [p.pt for p in k2], desc])

    return all_matches
    