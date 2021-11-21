from collections import defaultdict
import os
import shutil
import subprocess
import cv2

# Indicate the openMVG binary directory
OPENMVG_SFM_BIN = "/home/aswerdlow/github/openmvg_build/Linux-x86_64-RELEASE"

# Indicate the openMVG camera sensor width directory
CAMERA_SENSOR_WIDTH_DIRECTORY = "/home/aswerdlow/github/openMVG/src/software/SfM" + "/../../openMVG/exif/sensor_width_database"

input_dir = 'tmp_data/images'
output_dir = 'tmp_data/output'
matches_dir = os.path.join(output_dir, "matches")
reconstruction_dir = os.path.join(output_dir, "reconstruction_sequential")
camera_file_params = os.path.join(CAMERA_SENSOR_WIDTH_DIRECTORY, "sensor_width_camera_database.txt")


def get_matches(cur_frame, panos):
    if os.path.exists(input_dir):
        shutil.rmtree(input_dir)
    os.makedirs(input_dir)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    if os.path.exists(matches_dir):
        shutil.rmtree(matches_dir)
    os.makedirs(matches_dir)

    frame_id, frame = cur_frame
    frame_file = os.path.abspath(f'{input_dir}/frame-{frame_id}.jpg')
    cv2.imwrite(frame_file, cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
    for pano_id, (_, pano_frame) in panos.items():
        pano_file = os.path.abspath(f'{input_dir}/{pano_id}.jpg')
        cv2.imwrite(pano_file, cv2.cvtColor(pano_frame, cv2.COLOR_RGB2GRAY))

    subprocess.run([os.path.join(OPENMVG_SFM_BIN, "openMVG_main_SfMInit_ImageListing"),  "-i", input_dir, "-o",
                    matches_dir, "-d", camera_file_params], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    import time
    start_time = time.time()
    subprocess.run( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeFeatures"),  "-i", matches_dir+"/sfm_data.json", "-o", matches_dir, "-m", "SIFT"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # subprocess.run([os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeFeatures_OpenCV"),  "-i", matches_dir+"/sfm_data.json","-o", matches_dir, "-m", "SIFT_OPENCV"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    subprocess.run([os.path.join(OPENMVG_SFM_BIN, "openMVG_main_PairGenerator"), "-i", matches_dir+"/sfm_data.json",
                    "-o", matches_dir + "/pairs.bin"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    subprocess.run([os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeMatches"),  "-i", matches_dir + "/sfm_data.json", "-p", matches_dir +
                    "/pairs.bin", "-o", matches_dir + "/matches.putative.bin"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    subprocess.run([os.path.join(OPENMVG_SFM_BIN, "openMVG_main_GeometricFilter"), "-i", matches_dir+"/sfm_data.json",           "-m", matches_dir +
                    "/matches.putative.bin", "-g", "f", "-o", matches_dir+"/matches.f.bin"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    pExport = subprocess.run([os.path.join(OPENMVG_SFM_BIN, "openMVG_main_exportMatches"), "-i", matches_dir+"/sfm_data.json", "-m", matches_dir +
                    "/matches.f.bin", "-o", output_dir+"/export", '-d', matches_dir], stdout=subprocess.PIPE, text=True, stderr=subprocess.DEVNULL).stdout.splitlines()

    matches = list(map(lambda a: a[2:].rstrip().split(','), filter(lambda x: x.startswith("m:") and 'frame' in x, pExport)))
    keypoint_dict = {}
    for n1, x1, y1, n2, x2, y2 in matches:
        pano_file_name = n1 if 'frame' in n2 else n2
        if pano_file_name not in keypoint_dict:
            keypoint_dict[pano_file_name] = [[], [], []]

        keypoint_dict[pano_file_name][0].append(cv2.KeyPoint(float(x1), float(y1), 1))
        keypoint_dict[pano_file_name][1].append(cv2.KeyPoint(float(x2), float(y2), 1))
        keypoint_dict[pano_file_name][2].append([cv2.DMatch(len(keypoint_dict[pano_file_name][2]) + 1, len(keypoint_dict[pano_file_name][2]) + 1, 1)])

    for pano_file_name, (k1, k2, desc) in keypoint_dict.items():
        reference_img = cv2.drawMatchesKnn(frame, k1, pano_frame, k2, desc, None, flags=2)
        cv2.imwrite(f'tmp-reg/{frame_id}+{pano_file_name[:-4]}.jpg', reference_img)

    return [(panos[pano_file_name[:-4]], k1, k2, desc) for pano_file_name, (k1, k2, desc) in keypoint_dict.items()]
    # tot_time = time.time()-start_time
    # print(tot_time)
    # 86608 matches found, 1.2389135360717773, opencv
    # 72409 matches found, 4.997808218002319
