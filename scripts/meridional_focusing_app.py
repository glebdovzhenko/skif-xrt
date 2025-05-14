"""
## Caciuffo Melone Rustichelli Boeuf
$$
R_s = \frac{2pq \cdot \sin \theta}{(p+q)}
$$
$$
R_m = \frac{pq}{R_s \cdot \cos \theta} \sqrt{1 - R_s^2 / pq}
$$
"""

import numpy as np
import plotly.graph_objects as go
from copy import deepcopy


def ccw(A, B, C):
    """
    Checks if 3 points A, B, C are listed in counterclockwise order
    """
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A, B, C, D):
    """
    Checks if AB intersects with CD. Runs into problems if the two are parallel
    """
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    result = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    v1_u = np.pad(v1_u, (0, 1))
    v2_u = np.pad(v2_u, (0, 1))
    return result * np.sign(np.dot(np.cross(v1_u, v2_u), np.array([0, 0, 1])))


class CrystalFocusingScene:
    """"""

    # general
    fig_size = 600
    # plot styling
    crystal_kwargs = {
        "marker": {"size": 0, "opacity": 0},
        "line": {"width": 1, "color": "purple"},
        "showlegend": False,
    }
    source_kwargs = {
        "marker": {"size": 5, "color": "white"},
        "line": {"width": 0},
        "showlegend": False,
    }
    rowland_circle_kwargs = {
        "marker": {"size": 0, "opacity": 0},
        "line": {"width": 1, "dash": "dash", "color": "lightgray"},
        "name": "Rowland circle",
    }
    focal_circle_kwargs = {
        "marker": {"size": 0, "opacity": 0},
        "line": {"width": 1, "dash": "dash", "color": "gray"},
        "name": "Focal circle",
    }
    inc_rays_kwargs = {
        "marker": {"size": 0, "opacity": 0},
        "line": {"width": 1, "color": "white"},
        "showlegend": False,
    }
    surface_norms_kwargs = {
        "marker": {"size": 0, "opacity": 0},
        "line": {"width": 1, "color": "purple"},
        "showlegend": False,
    }
    refl_norms_kwargs = {
        "marker": {"size": 0, "opacity": 0},
        "line": {"width": 1, "color": "black"},
        "showlegend": False,
    }
    refl_rays_kwargs = {
        "marker": {"size": 0, "opacity": 0},
        "line": {"width": 1, "color": "white"},
        "showlegend": False,
    }
    refl_int_kwargs = {
        "mode": "markers",
        "marker": {"size": 5, "color": "red", "opacity": 1},
        "line": {"width": 0},
        "showlegend": True,
        "name": "Image (rays)",
    }
    caciuffo_kwargs = {
        "mode": "markers",
        "marker": {"size": 5, "color": "MediumPurple", "opacity": 1},
        "line": {"width": 0},
        "showlegend": True,
        "name": "Image (Caciuffo)",
    }
    _data = {
        "crystal": {
            "type": "circle",  # "circle", "parabola"
            "R": 10.0,  # for a circle R is radius, for a parabola focus
            "anglim": [-10.0, 10.0],  # degrees
            "npts": 2000,
            "chi": 10.0,  # degrees, asymmetric cut angle
        },
        "source": {
            "distance": 10.0,  # mm
            "angle": 110.0,  # degrees
            "divergence": 3.0,  # degrees
            "nrays": 10,
        },
        "scene": {
            "surf_norms": True,
            "refl_norms": True,
        },
    }

    def __init__(self):
        self.data = self._data
        self.src_xy = None
        self.cr_points = None
        self.cr_surface_norms = None
        self.cr_refl_norms = None
        self.source_rays_initial = None
        self.source_rays_endpoints = None
        self.reflected_rays_exist = False
        self.reflected_rays_initial = None
        self.reflected_rays_intersections = None
        self.mean_image_distance = None
        self.reflected_rays_endpoints = None
        self.theta = None

    def clear_computation(self):
        self.src_xy = None
        self.cr_points = None
        self.cr_surface_norms = None
        self.cr_refl_norms = None
        self.source_rays_initial = None
        self.source_rays_endpoints = None
        self.reflected_rays_exist = False
        self.reflected_rays_initial = None
        self.reflected_rays_intersections = None
        self.mean_image_distance = None
        self.reflected_rays_endpoints = None
        self.theta = None

    def compute(self):
        # source xy coordinates
        self.src_xy = np.array(
            [
                self.data["source"]["distance"]
                * np.cos(np.radians(self.data["source"]["angle"])),
                self.data["source"]["distance"]
                * np.sin(np.radians(self.data["source"]["angle"])),
            ]
        )

        # xy coordinates of the crystal surface
        if self.data["crystal"]["type"] == "circle":
            cphs = (
                np.radians(
                    np.linspace(
                        *self.data["crystal"]["anglim"], self.data["crystal"]["npts"]
                    )
                )
                - np.pi / 2.0
            )
            self.cr_points = np.array(
                [
                    self.data["crystal"]["R"] * np.cos(cphs),
                    self.data["crystal"]["R"] * (1.0 + np.sin(cphs)),
                ]
            )
        elif self.data["crystal"]["type"] == "parabola":
            raise NotImplementedError()
        else:
            raise ValueError()

        # Incident rays endpoints crossing the crystal surface
        central_direction = np.radians(self.data["source"]["angle"]) - np.pi
        le_direction = (
            central_direction - np.radians(self.data["source"]["divergence"]) / 2
        )
        he_direction = (
            central_direction + np.radians(self.data["source"]["divergence"]) / 2
        )
        r = 2.0 * self.data["source"]["distance"]
        cphs = np.linspace(le_direction, he_direction, self.data["source"]["nrays"])
        self.source_rays_initial = np.array(
            [r * np.cos(cphs), r * np.sin(cphs)]
        ) + np.repeat(self.src_xy[:, np.newaxis], self.data["source"]["nrays"], axis=1)

        # Incident rays endpoints on the crystal surface
        src = self.src_xy
        rays = self.source_rays_initial.T
        cr_points = self.cr_points.T
        rays_endpoints = []
        rays_endpoints_ids = []
        for ii in range(rays.shape[0]):
            for jj in range(cr_points.shape[0] - 1):
                if intersect(src, rays[ii], cr_points[jj], cr_points[jj + 1]):
                    rays_endpoints.append(cr_points[jj])
                    rays_endpoints_ids.append(jj)
                    break
        self.source_rays_endpoints = np.array(rays_endpoints).T
        self.rays_endpoints_ids = np.array(rays_endpoints_ids).astype(int)

        # Local surface norms of the crystal
        if self.data["crystal"]["type"] == "circle":
            cphs = (
                np.radians(
                    np.linspace(
                        *self.data["crystal"]["anglim"], self.data["crystal"]["npts"]
                    )
                )
                - np.pi / 2.0
            )
            self.cr_surface_norms = np.array([-np.cos(cphs), -np.sin(cphs)])
        elif self.data["crystal"]["type"] == "parabola":
            raise NotImplementedError()
        else:
            raise ValueError()
        self.cr_surface_norms = self.cr_surface_norms.T[self.rays_endpoints_ids].T

        # Rotation matrix between surface and reflecting planes
        mrot = np.array(
            [
                [
                    np.cos(np.radians(self.data["crystal"]["chi"])),
                    -np.sin(np.radians(self.data["crystal"]["chi"])),
                ],
                [
                    np.sin(np.radians(self.data["crystal"]["chi"])),
                    np.cos(np.radians(self.data["crystal"]["chi"])),
                ],
            ]
        )

        # Local reflecting plane norms of the crystal
        if self.data["crystal"]["type"] == "circle":
            self.cr_refl_norms = self.cr_surface_norms.T
            self.cr_refl_norms = np.array([mrot @ pt for pt in self.cr_refl_norms]).T
        elif self.data["crystal"]["type"] == "parabola":
            raise NotImplementedError()
        else:
            raise ValueError()

        # reflected rays
        inc_rays_directions = np.array(
            [self.src_xy - pt for pt in self.source_rays_endpoints.T]
        ).T
        angles = np.array(
            [
                angle_between(v1, v2)
                for v1, v2 in zip(inc_rays_directions.T, self.cr_refl_norms.T)
            ]
        )
        if np.any(angles >= (np.pi / 2)) or np.any(angles <= (-np.pi / 2)):
            self.reflected_rays_exist = False
            self.reflected_rays_initial = np.array([[0.0, 0.0] for _ in angles]).T
        else:
            self.reflected_rays_exist = True
            self.reflected_rays_initial = np.array(
                [
                    np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
                    @ vec
                    for vec, ang in zip(inc_rays_directions.T, 2.0 * angles)
                ]
            ).T

        # intersections of reflected rays
        if self.reflected_rays_exist:
            reflected_rays = [
                [start, end]
                for start, end in zip(
                    self.source_rays_endpoints.T, self.reflected_rays_initial.T
                )
            ]
            self.reflected_rays_intersections = []
            for ii in range(len(reflected_rays)):
                for jj in range(ii + 1, len(reflected_rays)):
                    ray1 = deepcopy(reflected_rays[ii])
                    ray2 = deepcopy(reflected_rays[jj].copy())
                    ray1[1] += ray1[0]
                    ray2[1] += ray2[0]
                    # y = kx + b
                    r1k = (ray1[1][1] - ray1[0][1]) / (ray1[1][0] - ray1[0][0])
                    r1b = ray1[0][1] - ray1[0][0] * (ray1[1][1] - ray1[0][1]) / (
                        ray1[1][0] - ray1[0][0]
                    )
                    r2k = (ray2[1][1] - ray2[0][1]) / (ray2[1][0] - ray2[0][0])
                    r2b = ray2[0][1] - ray2[0][0] * (ray2[1][1] - ray2[0][1]) / (
                        ray2[1][0] - ray2[0][0]
                    )
                    try:
                        self.reflected_rays_intersections.append(
                            np.linalg.solve(
                                a=np.array([[-r1k, 1], [-r2k, 1]]),
                                b=np.array([r1b, r2b]),
                            )
                        )
                    except np.linalg.LinAlgError:
                        pass
            if self.reflected_rays_intersections:
                self.reflected_rays_intersections = np.array(
                    self.reflected_rays_intersections
                ).T
                self.mean_image_distance = np.mean(
                    np.apply_along_axis(
                        np.linalg.norm, 0, self.reflected_rays_intersections
                    )
                )
            else:
                self.reflected_rays_intersections = np.array([[], []])
                self.mean_image_distance = 0
        else:
            self.reflected_rays_intersections = np.array([[], []])
            self.mean_image_distance = 0

        # endpoints for reflected rays based on the image distance
        if self.reflected_rays_exist:
            if self.reflected_rays_intersections.shape[1] != 0:
                length = 1.2 * np.max(
                    np.apply_along_axis(
                        np.linalg.norm, 0, self.reflected_rays_intersections
                    )
                )
            else:
                length = 1.2 * self.data["source"]["distance"]
            self.reflected_rays_endpoints = (
                length
                * np.array(
                    [pt / np.linalg.norm(pt) for pt in self.reflected_rays_initial.T]
                ).T
            )
        else:
            self.reflected_rays_endpoints = np.array([[0.0, 0.0] for _ in angles]).T

        # Bragg angle for output
        if self.reflected_rays_exist:
            self.theta = (
                180 - self.data["source"]["angle"] + self.data["crystal"]["chi"]
            )
        else:
            self.theta = 0.0

    @property
    def caciuffo_image_location(self):
        """
        This formula is simple:
        2 / R = sin(theta + alpha) / p + sin(Th - alpha) / q
        alpha is the angle between lattice planes and crystal surface,
        th + alpha and th - alpha are glancing angles of incidence and emergence

        I think (!) my self.data["source"]["angle"] is theta + alpha in this notation
        and my self.data["crystal"]["chi"] is -alpha.
        """
        if 0 <= self.data["source"]["angle"] <= 90:
            theta = np.pi - np.radians(self.data["source"]["angle"])
        elif 90 < self.data["source"]["angle"] <= 180:
            theta = np.pi - np.radians(self.data["source"]["angle"])
        else:
            raise ValueError()

        alpha = -np.radians(self.data["crystal"]["chi"])
        theta -= alpha
        p = self.data["source"]["distance"]
        q = np.sin(theta - alpha) / (
            2.0 / self.data["crystal"]["R"] - np.sin(theta + alpha) / p
        )
        init_k = -self.src_xy.copy() / p
        ref_k = (
            np.array(
                [
                    [np.cos(2.0 * theta), -np.sin(2.0 * theta)],
                    [np.sin(2.0 * theta), np.cos(2.0 * theta)],
                ]
            )
            @ init_k
        )
        ref_k *= q
        return ref_k

    def crystal_plot_data(self):
        xs, ys = self.cr_points
        return {"x": xs, "y": ys}

    def rowland_circle_plot_data(self):
        cphs = np.linspace(0, 2.0 * np.pi, 1000)
        xs, ys = 0.5 * self.data["crystal"]["R"] * np.cos(cphs), 0.5 * self.data[
            "crystal"
        ]["R"] * (1.0 + np.sin(cphs))
        return {"x": xs, "y": ys}

    def focal_circle_plot_data(self):
        cphs = np.linspace(0, 2.0 * np.pi, 1000)
        xs, ys = 0.25 * self.data["crystal"]["R"] * np.cos(cphs), 0.25 * self.data[
            "crystal"
        ]["R"] * (1.0 + np.sin(cphs))
        return {"x": xs, "y": ys}

    def source_plot_data(self):
        x, y = self.src_xy
        return {"x": [x], "y": [y]}

    def inc_rays_plot_data(self):
        x_s, y_s = self.src_xy
        return [{"x": [x_s, x], "y": [y_s, y]} for x, y in self.source_rays_endpoints.T]

    def surface_norms_plot_data(self):
        return [
            {"x": [p1[0], p1[0] + p2[0]], "y": [p1[1], p1[1] + p2[1]]}
            for p1, p2 in zip(self.source_rays_endpoints.T, self.cr_surface_norms.T)
        ]

    def refl_norms_plot_data(self):
        return [
            {"x": [p1[0], p1[0] + p2[0]], "y": [p1[1], p1[1] + p2[1]]}
            for p1, p2 in zip(self.source_rays_endpoints.T, self.cr_refl_norms.T)
        ]

    def refl_rays_plot_data(self):
        return [
            {"x": [p1[0], p1[0] + p2[0]], "y": [p1[1], p1[1] + p2[1]]}
            for p1, p2 in zip(
                self.source_rays_endpoints.T, self.reflected_rays_endpoints.T
            )
        ]

    def refl_int_plot_data(self):
        return {
            "x": self.reflected_rays_intersections[0],
            "y": self.reflected_rays_intersections[1],
        }

    def caciuffo_plot_data(self):
        x, y = self.caciuffo_image_location
        return {"x": [x], "y": [y]}

    def build_plot(self):
        fig = go.Figure()
        fig.update_layout(
            width=self.fig_size,
            height=self.fig_size,
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1, showgrid=False)
        fig.update_xaxes(showgrid=False)
        fig.add_trace(go.Scatter(**self.crystal_plot_data(), **self.crystal_kwargs))
        fig.add_trace(go.Scatter(**self.source_plot_data(), **self.source_kwargs))
        fig.add_trace(
            go.Scatter(**self.rowland_circle_plot_data(), **self.rowland_circle_kwargs)
        )
        fig.add_trace(
            go.Scatter(**self.focal_circle_plot_data(), **self.focal_circle_kwargs)
        )
        for kwds in self.inc_rays_plot_data():
            fig.add_trace(go.Scatter(**kwds, **self.inc_rays_kwargs))
        if self.data["scene"]["surf_norms"]:
            for kwds in self.surface_norms_plot_data():
                fig.add_trace(go.Scatter(**kwds, **self.surface_norms_kwargs))
        if self.data["scene"]["refl_norms"]:
            for kwds in self.refl_norms_plot_data():
                fig.add_trace(go.Scatter(**kwds, **self.refl_norms_kwargs))
        for kwds in self.refl_rays_plot_data():
            fig.add_trace(go.Scatter(**kwds, **self.refl_rays_kwargs))
        fig.add_trace(go.Scatter(**self.refl_int_plot_data(), **self.refl_int_kwargs))
        fig.add_trace(go.Scatter(**self.caciuffo_plot_data(), **self.caciuffo_kwargs))
        return fig


if __name__ == "__main__":
    from dash import Dash, html, dcc, callback, Output, Input

    scene = CrystalFocusingScene()

    app = Dash()
    app.layout = html.Div(
        [
            dcc.Graph(id="graph-content"),
            html.Div(
                [
                    html.Label("Crystal"),
                    html.Div(
                        [
                            html.Label("Radius"),
                            dcc.Input(
                                id="in-cr-radius",
                                type="number",
                                value=scene.data["crystal"]["R"],
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "row"},
                    ),
                    html.Div(
                        [
                            html.Label("Min angle"),
                            dcc.Input(
                                id="in-cr-angmin",
                                type="number",
                                value=scene.data["crystal"]["anglim"][0],
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "row"},
                    ),
                    html.Div(
                        [
                            html.Label("Max angle"),
                            dcc.Input(
                                id="in-cr-angmax",
                                type="number",
                                value=scene.data["crystal"]["anglim"][1],
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "row"},
                    ),
                    html.Div(
                        [
                            html.Label("Points"),
                            dcc.Input(
                                id="in-cr-npts",
                                type="number",
                                value=scene.data["crystal"]["npts"],
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "row"},
                    ),
                    html.Div(
                        [
                            html.Label("Asymmetry"),
                            dcc.Input(
                                id="in-cr-chi",
                                type="number",
                                value=scene.data["crystal"]["chi"],
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "row"},
                    ),
                    html.Label("Source"),
                    html.Div(
                        [
                            html.Label("Distance"),
                            dcc.Input(
                                id="in-s-dist",
                                type="number",
                                value=scene.data["source"]["distance"],
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "row"},
                    ),
                    html.Div(
                        [
                            html.Label("Angle"),
                            dcc.Input(
                                id="in-s-ang",
                                type="number",
                                value=scene.data["source"]["angle"],
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "row"},
                    ),
                    html.Div(
                        [
                            html.Label("Divergence"),
                            dcc.Input(
                                id="in-s-div",
                                type="number",
                                value=scene.data["source"]["divergence"],
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "row"},
                    ),
                    html.Div(
                        [
                            html.Label("N rays"),
                            dcc.Input(
                                id="in-s-nrays",
                                type="number",
                                value=scene.data["source"]["nrays"],
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "row"},
                    ),
                    html.Div(
                        [
                            html.Label("Scene"),
                            dcc.Checklist(
                                ["Surface norms", "Reflection norms"],
                                ["Surface norms", "Reflection norms"],
                                id="scene-checklist",
                            ),
                        ],
                        style={
                            "display": "flex",
                            "flexDirection": "column",
                            "textAlign": "center",
                        },
                    ),
                    html.Label("Source Distance", id="source-dist-lbl"),
                    html.Label("Mean Image Distance", id="mean-image-lbl"),
                    html.Label("Analytical Image Distance", id="an-image-lbl"),
                    html.Label("Bragg angle", id="bragg-angle-lbl"),
                ],
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "textAlign": "center",
                },
            ),
        ],
        style={"display": "flex", "flexDirection": "row"},
    )

    @callback(
        Output("graph-content", "figure"),
        Output("mean-image-lbl", "children"),
        Output("an-image-lbl", "children"),
        Output("source-dist-lbl", "children"),
        Output("bragg-angle-lbl", "children"),
        Input("in-cr-radius", "value"),
        Input("in-cr-angmin", "value"),
        Input("in-cr-angmax", "value"),
        Input("in-cr-npts", "value"),
        Input("in-cr-chi", "value"),
        Input("scene-checklist", "value"),
        Input("in-s-dist", "value"),
        Input("in-s-ang", "value"),
        Input("in-s-div", "value"),
        Input("in-s-nrays", "value"),
    )
    def update_plot(
        cr_radius,
        cr_angmin,
        cr_angmax,
        cr_npts,
        cr_chi,
        scene_checklist,
        source_dist,
        source_ang,
        source_div,
        source_nrays,
    ):
        if cr_radius is not None:
            scene.data["crystal"]["R"] = cr_radius
        if (cr_angmin is not None) and (cr_angmax is not None):
            scene.data["crystal"]["anglim"] = [cr_angmin, cr_angmax]
        if isinstance(cr_npts, int) and cr_npts > 0:
            scene.data["crystal"]["npts"] = cr_npts
        if cr_chi is not None:
            scene.data["crystal"]["chi"] = cr_chi

        scene.data["scene"]["surf_norms"] = "Surface norms" in scene_checklist
        scene.data["scene"]["refl_norms"] = "Reflection norms" in scene_checklist

        if source_dist is not None:
            scene.data["source"]["distance"] = source_dist
        if source_ang is not None:
            scene.data["source"]["angle"] = source_ang
        if source_div is not None:
            scene.data["source"]["divergence"] = source_div
        if isinstance(source_nrays, int) and source_nrays > 0:
            scene.data["source"]["nrays"] = source_nrays

        scene.clear_computation()
        scene.compute()
        return (
            scene.build_plot(),
            f"Mean Image Distance: {scene.mean_image_distance:.2f}",
            f"Analytical Image Distance: {np.linalg.norm(scene.caciuffo_image_location):.2f}",
            f"Source Distance: {scene.data["source"]["distance"]:.2f}",
            f"Bragg angle: {scene.theta:.2f}",
        )

    app.run(debug=True)
