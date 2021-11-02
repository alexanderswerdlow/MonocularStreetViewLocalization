
from typing import NamedTuple
from os import listdir
from os.path import isfile, join, splitext
import pickle
from config import images_dir

class Loc(NamedTuple):
    lat: float
    long: float


class Pano(NamedTuple):
    loc: Loc
    pano_id: str
    depth_map: dict = None

    def __hash__(self):
        return hash(self.pano_id)

    def __eq__(self, other):
        return isinstance(other, Pano) and self.pano_id == other.pano_id

    def get_name(self):
        return self.pano_id


def separate_loc_list(input_list: list[Loc]) -> tuple[list[float], list[float]]:
    return tuple(zip(*input_list))


def combine_lat_long_lists(lat_list: list[float], long_list: list[float]) -> list[Loc]:
    if len(lat_list) != len(long_list):
        raise Exception("List lengths are different")
    else:
        return list(map(lambda x, y: Loc(x, y), lat_list, long_list))


def get_existing_panoramas() -> set[Pano]:
    try:
        existing_panoramas = pickle.load(open(f"{images_dir}/meta.p", "rb"))
    except (OSError, IOError) as e:
        existing_panoramas = set()

    mypath = f'{images_dir}/'
    onlyfiles = set([splitext(f)[0] for f in listdir(mypath) if isfile(join(mypath, f)) and f != '.DS_Store' and f != 'meta.p'])
    for p in existing_panoramas:
        if p.pano_id not in onlyfiles:
            existing_panoramas.remove(p)

    return existing_panoramas


def save_existing_panoramas(panos):
    pickle.dump(panos, open(f"{images_dir}/meta.p", "wb"))
