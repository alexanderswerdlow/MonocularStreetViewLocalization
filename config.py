from os import environ, getenv
if environ.get('STREET_VIEW_DATA_DIR') is not None:
    data_dir = getenv("STREET_VIEW_DATA_DIR")
else:
    data_dir = '../data' # Put your data directory here
images_dir = f'{data_dir}/images'
sqlite_path = f'{data_dir}/street_view_cache'

start_frame = 5000
segmentation_model_dir = './localization/enet-cityscapes'

recording_dir = f'{data_dir}/recordings/2021-11-10T13-16-47'
use_pickled_images = False