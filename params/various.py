import numpy as np

si_compliance_tensor = np.array([
    [ 8.7, -2.3, -2.3, 0.,   0.,   0.],
    [-2.3,  8.7, -2.3, 0.,   0.,   0.],
    [-2.3, -2.3,  8.7, 0.,   0.,   0.],
    [ 0.,   0.,   0.,  13.4, 0.,   0.],
    [ 0.,   0.,   0.,  0.,   13.4, 0.],
    [ 0.,   0.,   0.,  0.,   0.,   13.4],
])  # GPa, https://next-gen.materialsproject.org/materials/mp-149

si_poisson_ratio = .27
