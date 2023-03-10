import numpy as np

front_end_distance = 15000  # from source
front_end_h_angle = 2.0e-3  # rad
front_end_v_angle = 0.2e-3  # rad
front_end_opening = [
    -front_end_distance * np.tan(front_end_h_angle / 2.),
    front_end_distance * np.tan(front_end_h_angle / 2.),
    -front_end_distance * np.tan(front_end_v_angle / 2.),
    front_end_distance * np.tan(front_end_v_angle / 2.)
]

filter_distance = 18000  # from source

monochromator_distance = 33500  # from source
monochromator_z_offset = 25  # fixed beam offset in z direction
monochromator_x_lim = [-100., 100.]  # crystal surface area: min, max x in local coordinates
monochromator_y_lim = [-10., 10.]  # crystal surface area: min, max y in local coordinates

crl_distance = 55000.  # from source

