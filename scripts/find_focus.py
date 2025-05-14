import numpy as np
import pymc
import pickle

import xarray as xr

import arviz as az

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from matplotlib import pyplot as plt

intensity_cutoff = 1e-3

with open(
    "/home/glebd/Dev/skif-xrt/beamlines/xml/01_sagittal_dcm/dcm_60keV_frontend_fz.npy",
    "br",
) as f:
    data = pickle.load(f)
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "scatter3d"}]],
    )
    xbinCenters = (data.xbinEdges[1:] + data.xbinEdges[:-1]) / 2.0
    ybinCenters = (data.ybinEdges[1:] + data.ybinEdges[:-1]) / 2.0
    fig.add_trace(
        go.Surface(z=data.total2D, x=xbinCenters, y=ybinCenters), row=1, col=2
    )

    pts_coord = []
    pts_direction = []
    pts_intensity = []
    for ii in range(data.total2D.shape[0]):
        for jj in range(data.total2D.shape[1]):
            if data.total2D[ii, jj] >= intensity_cutoff:
                pts_coord.append(xbinCenters[ii])
                pts_direction.append(ybinCenters[jj])
                pts_intensity.append(data.total2D[ii, jj])

    pts_coord = np.array(pts_coord)
    pts_direction = np.array(pts_direction)
    pts_intensity = np.array(pts_intensity)

    fig.add_trace(
        go.Scatter(
            x=pts_direction,
            y=pts_coord,
            mode="markers",
            marker={"color": pts_intensity},
        ),
        row=1,
        col=1,
    )
    fig.show()

    with pymc.Model() as model:
        #
        #
        # Our model is:
        # z = Y_s * z' - Z_s
        #
        #
        # Define priors
        sigma = pymc.HalfCauchy("Sigma", beta=1e5)
        intercept = pymc.Normal("Zs", 0.0, sigma=1e5)
        slope = pymc.Normal("Ys", 0.0, sigma=1e5)

        weights = pts_intensity / pts_intensity.max()
        scaled_sigma = pymc.Deterministic("scaled_sigma", sigma / weights)

        # Define likelihood
        likelihood = pymc.Normal(
            "crd",
            mu=-intercept + slope * pts_direction,
            # sigma=sigma,
            sigma=scaled_sigma,
            observed=pts_coord,
        )

        # Inference!
        idata = pymc.sample(10000)

        # Compute posteriors
        idata.posterior["Fdist"] = (
            idata.posterior["Zs"] ** 2 + idata.posterior["Ys"] ** 2
        ) ** 0.5
        idata.posterior["crd_model"] = -idata.posterior["Zs"] + idata.posterior[
            "Ys"
        ] * xr.DataArray(pts_direction)

        # Plot results
        az.plot_trace(idata, var_names=["Ys", "Zs", "Fdist", "Sigma"])
        plt.tight_layout()

        _, ax = plt.subplots(figsize=(7, 7))
        az.plot_lm(
            idata=idata,
            y="crd",
            x=xr.DataArray(pts_direction),
            num_samples=100,
            axes=ax,
            y_model="crd_model",
        )
        ax.set_title("Posterior predictive regression lines")
        ax.set_xlabel("x")

        plt.show()
