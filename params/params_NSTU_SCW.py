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

filter_distance = 23000  # from source
diamond_filter_th = .5  # mm
diamond_filter_N = 15
sic_filter_th = .5
sic_filter_N = 7
filter_size_z = 10.  # mm
filter_size_x = 50.  # mm

monochromator_distance = 33500  # from source
monochromator_z_offset = 25  # fixed beam offset in z direction
monochromator_x_lim = [-100., 100.]  # crystal surface area: min, max x in local coordinates
monochromator_y_lim = [-10., 10.]  # crystal surface area: min, max y in local coordinates

crl_mask_distance = 27500.  # from source

croc_crl_distance = 28000.  # from source
# Be	
# croc_crl_y_t = 1.228
# croc_crl_L = 270
# # Al	
# croc_crl_y_t = 0.29
# croc_crl_L = 54
# # Dia	
# croc_crl_y_t = 0.58
# croc_crl_L = 50
# # Gr	
# croc_crl_y_t = 0.58
# croc_crl_L = 81
# glassy carbon	
croc_crl_y_t = 0.85
croc_crl_L = 230.
croc_crl_N = 115

exit_slit_distance = 56000.  # from source

