import numpy as np

front_end_distance = 15000.0  # from source
front_end_h_angle = 2.0e-3  # rad
front_end_v_angle = 0.2e-3  # rad
front_end_opening = [
    -front_end_distance * np.tan(front_end_h_angle / 2.0),
    front_end_distance * np.tan(front_end_h_angle / 2.0),
    -front_end_distance * np.tan(front_end_v_angle / 2.0),
    front_end_distance * np.tan(front_end_v_angle / 2.0),
]

filter_distance = 18000.0  # from source

filter1_distance = 17700.0
filter2_distance = 18400.0
filter3_distance = 19100.0
filter4_distance = 22100.0
filter5_distance = 25600.0

diamond_filter_thickness = 0.5
sic_filter_thickness = 0.35

monochromator_distance = 33500.0  # from source
monochromator_z_offset = 25.0  # fixed beam offset in z direction
monochromator_x_lim = [
    -50.0,
    50.0,
]  # crystal surface area: min, max x in local coordinates
monochromator_y_lim = [
    -10.0,
    10.0,
]  # crystal surface area: min, max y in local coordinates

focusing_distance = 51000.0  # from source

exit_slit_distance = 115000.0  # from source
exit_slit_opening = [-107.5, 107.5, -10.0, 10.0]
