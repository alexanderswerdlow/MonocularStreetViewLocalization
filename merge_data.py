from config import data_dir
import pickle
import copyreg
import cv2

def _pickle_dmatch(dmatch):
    return cv2.DMatch, (dmatch.queryIdx, dmatch.trainIdx, dmatch.imgIdx, dmatch.distance)


copyreg.pickle(cv2.DMatch().__class__, _pickle_dmatch)

files_to_open = ['7200']
saved_matches = []
final_matches = {}
for f in files_to_open:
    print(f'Starting on {f}')
    try:
        final_matches = {**final_matches, **pickle.load(open(f"{data_dir}/kvld_matches_{f}.p", "rb"))}
    except (OSError, IOError) as e:
        print(f"Failed to read: {f}")


pickle.dump(final_matches, open(f"{data_dir}/kvld_matches_merged_2.p", "wb"))

# import pickle
# from config import data_dir
# final_matches = pickle.load(open(f"{data_dir}/estimated_output_ceres.p", "rb"))
# print(final_matches)