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
    frame_file = os.path.abspath(f'{input_dir}/{frame_id}.jpg')
    cv2.imwrite(frame_file, frame)
    for pano_id, pano_frame  in panos:
        pano_file = os.path.abspath(f'{input_dir}/{pano_id}.jpg')
        cv2.imwrite(pano_file, pano_frame)

    print ("1. Intrinsics analysis")
    pIntrisics = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_SfMInit_ImageListing"),  "-i", input_dir, "-o", matches_dir, "-d", camera_file_params] )
    pIntrisics.wait()

    print ("2. Compute features")
    import time; start_time = time.time()
    # pFeatures = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeFeatures"),  "-i", matches_dir+"/sfm_data.json", "-o", matches_dir, "-m", "SIFT"] )
    pFeatures = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeFeatures_OpenCV"),  "-i", matches_dir+"/sfm_data.json", "-o", matches_dir, "-m", "SIFT_OPENCV"] )
    pFeatures.wait()
    tot_time = time.time()-start_time

    print ("3. Compute matching pairs")
    pPairs = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_PairGenerator"), "-i", matches_dir+"/sfm_data.json", "-o" , matches_dir + "/pairs.bin" ] )
    pPairs.wait()

    print ("4. Compute matches")
    pMatches = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeMatches"),  "-i", matches_dir+"/sfm_data.json", "-p", matches_dir+ "/pairs.bin", "-o", matches_dir + "/matches.putative.bin" ] )
    pMatches.wait()

    print ("5. Filter matches" )
    pFiltering = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_GeometricFilter"), "-i", matches_dir+"/sfm_data.json", "-m", matches_dir+"/matches.putative.bin" , "-g" , "f" , "-o" , matches_dir+"/matches.f.bin" ] )
    pFiltering.wait()

    print ("6. Export matches" )
    pExport = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_exportMatches"), "-i", matches_dir+"/sfm_data.json", "-m", matches_dir+"/matches.f.bin" , "-o" , output_dir+"/export", '-d', matches_dir ], stdout=subprocess.PIPE, encoding='utf8')
    pExport.wait()
    matches = list(map(lambda a: a[2:].rstrip().split(','), filter(lambda x: x.startswith("m:"), pExport.stdout.readlines())))
    print(matches)
    exit()

    print(tot_time)
    # 86608 matches found, 1.2389135360717773, opencv
    # 72409 matches found, 4.997808218002319


