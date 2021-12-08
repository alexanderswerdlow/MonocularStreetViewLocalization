from save import Vehicle
import os
import shutil

if __name__ == '__main__':

    import sys
    start_frame = int(sys.argv[1])

    # Delete tmp dir and recreate; Used for misc debug output
    dir = f'working/{start_frame}'
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)    

    test_vehicle = Vehicle(start_frame=start_frame)
    try:
        test_vehicle.iterate_frames()
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt")

