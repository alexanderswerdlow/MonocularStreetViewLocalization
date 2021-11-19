from vehicle import Vehicle
import os
import shutil

if __name__ == '__main__':

    # Delete tmp dir and recreate; Used for misc debug output
    dir = 'tmp'
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

    test_vehicle = Vehicle()
    try:
        test_vehicle.iterate_frames()
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt")
