from vehicle import Vehicle
import os
import shutil
from multiprocessing import get_context

def run_frames(solver):
    test_vehicle = Vehicle(solver=solver)
    test_vehicle.iterate_frames()
    print('finished', solver)
    return None

if __name__ == '__main__':
    # Delete tmp dir and recreate; Used for misc debug output
    dir = 'tmp'
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

    # Delete tmp dir and recreate; Used for misc debug output
    dir = 'tmp_data'
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
    
    multi = True

    if multi:
        try:
            with get_context("spawn").Pool(3) as pool:
                pool.map(run_frames, ['ceres', 'g2o', 'scipy'])
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt")
    else:
        run_frames('ceres')
    

