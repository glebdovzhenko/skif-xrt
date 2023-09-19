import numpy as np

front_end_distance = 15000.  # from source
front_end_h_angle = 2.0e-3  # rad
front_end_v_angle = 0.2e-3  # rad
front_end_opening = [
    -front_end_distance * np.tan(front_end_h_angle / 2.),
    front_end_distance * np.tan(front_end_h_angle / 2.),
    -front_end_distance * np.tan(front_end_v_angle / 2.),
    front_end_distance * np.tan(front_end_v_angle / 2.)
]

filter_distance = 18000.  # from source

filter1_distance = 17700.
filter2_distance = 18400.
filter3_distance = 19100.
filter4_distance = 22100.
filter5_distance = 25600.

diamond_filter_thickness = .5
sic_filter_thickness = .35

monochromator_distance = 33500.  # from source
monochromator_z_offset = 25.  # fixed beam offset in z direction
monochromator_x_lim = [-50., 50.]  # crystal surface area: min, max x in local coordinates
monochromator_y_lim = [-10., 10.]  # crystal surface area: min, max y in local coordinates

focusing_distance = 51000.  # from source

exit_slit_distance = 115000.  # from source
exit_slit_opening = [-107.5, 107.5, -10., 10.]

