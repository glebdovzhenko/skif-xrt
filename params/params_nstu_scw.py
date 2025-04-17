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

filter_distance = 23000  # from source
diamond_filter_th = 0.5  # mm
diamond_filter_N = 15
sic_filter_th = 0.5
sic_filter_N = 7
filter_size_z = 10.0  # mm
filter_size_x = 50.0  # mm

monochromator_distance = 33500  # from source
monochromator_z_offset = 25  # fixed beam offset in z direction
monochromator_x_lim = [
    -100.0,
    100.0,
]  # crystal surface area: min, max x in local coordinates
monochromator_y_lim = [
    -10.0,
    10.0,
]  # crystal surface area: min, max y in local coordinates

crl_mask_distance = 27500.0  # from source

croc_crl_distance = 28000.0  # from source
# not actually optimal
optimal_croc_geometry = {
    "Be": {"y_t": 0.7, "L": 110.0},  # Beryllium
    "Al": {"y_t": 0.3, "L": 55.0},  # Aluminium
    "Dia": {"y_t": 0.6, "L": 50.0},  # Diamond
    "GC": {"y_t": 0.6, "L": 115.0},  # Glassy carbon
}
# geometries that have 68, 87, and 95 percent of the absorption aperture
# covered by the physical aperture
croc_geometry_68_percent = {
    "Be": {"y_t": 0.5, "L": 130.0},  # Beryllium
    "Al": {"y_t": 0.2, "L": 60.0},  # Aluminium
    "GC": {"y_t": 0.4, "L": 120.0},  # Glassy carbon
}
croc_geometry_87_percent = {
    "Be": {"y_t": 0.7, "L": 280.0},  # Beryllium
    "Al": {"y_t": 0.3, "L": 70.0},  # Aluminium
    "GC": {"y_t": 0.6, "L": 250.0},  # Glassy carbon
}
croc_geometry_95_percent = {
    "Be": {"y_t": 0.9, "L": 470.0},  # Beryllium
    "Al": {"y_t": 0.4, "L": 120.0},  # Aluminium
    "GC": {"y_t": 0.8, "L": 440.0},  # Glassy carbon
}
exit_slit_distance = 56000.0  # from source
