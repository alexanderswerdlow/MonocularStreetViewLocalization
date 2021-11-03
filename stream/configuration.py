
FRAMES_CSV = 'Frames.txt'
GPS_CSV = 'GPS.txt'
MOTION_CSV = 'MotARH.txt'

FRAMES_DEFAULT = 30
GPS_DEFAULT    = 1
MOTION_DEFAULT = 100

FRAMES_FIELDS = [
    'frame_timestamp',
    'frame_number',
    'focal_length_x',
    'focal_length_y',
    'principal_point_x',
    'principal_point_y'
]

MOTION_FIELDS = [
    'motion_timestamp',
    'rotation_rate_x',
    'rotation_rate_y',
    'rotation_rate_z',
    'gravity_x',
    'gravity_y',
    'gravity_z',
    'user_accel_x',
    'user_accel_y',
    'user_accel_z',
    'motion_heading'
]

GPS_FIELDS = [
    'gps_timestamp',
    'latitude',
    'longitude',
    'horizontal_accuracy',
    'altitude',
    'vertical_accuracy',
    'floor_level',
    'course',
    'speed'
]