import numpy as np

front_end_distance = 15000  # from source
front_end_h_angle = 2.0e-3  # rad
front_end_v_angle = 0.2e-3  # rad
front_end_opening = [
    -front_end_distance * np.tan(front_end_h_angle / 2.0),
    front_end_distance * np.tan(front_end_h_angle / 2.0),
    -front_end_distance * np.tan(front_end_v_angle / 2.0),
    front_end_distance * np.tan(front_end_v_angle / 2.0),
]
