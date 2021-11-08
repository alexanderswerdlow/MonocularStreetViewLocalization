from os import environ, getenv
if environ.get('STREET_VIEW_DATA_DIR') is not None:
    data_dir = getenv("STREET_VIEW_DATA_DIR")
else:
    data_dir = '../data' # Put your data directory here
images_dir = f'{data_dir}/images'
sqlite_path = f'{data_dir}/street_view_cache'

skip_frames = 900