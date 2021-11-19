import pickle
from config import images_dir

def separate_loc_list(input_list):
    return tuple(zip(*input_list))


def combine_lat_long_lists(lat_list, long_list):
    if len(lat_list) != len(long_list):
        raise Exception("List lengths are different")
    else:
        return list(map(lambda x, y: (x, y), lat_list, long_list))

def get_existing_panoramas():
    try:
        existing_panoramas = pickle.load(open(f"{images_dir}/meta.p", "rb"))
    except (OSError, IOError) as e:
        existing_panoramas = set()

    return existing_panoramas

def save_existing_panoramas(panos):
    pickle.dump(panos, open(f"{images_dir}/meta.p", "wb"))