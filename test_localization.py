import signal

from vehicle import Vehicle
from config import recording_dir

if __name__ == '__main__':

    import os
    import shutil

    # Delete tmp dir and recreate; Used for misc debug output
    dir = 'tmp'
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

    sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    test_vehicle = Vehicle(recording_dir)
    signal.signal(signal.SIGINT, sigint_handler)
    try:
        test_vehicle.start(100)
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        test_vehicle.close()

    test_vehicle.close()