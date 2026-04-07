import numpy as np
import pickle

if __name__ == "__main__":
    fname = "/home/glebd/Dev/skif-xrt/beamlines/xml/NSTU_SCW/oh_power.npy"

    with open(fname, "rb") as data:
        data = pickle.load(data)

        # converts XRT units to W / mm^2
        power_density = data.total2D / (
            data.nRaysAll
            * np.mean(data.xbinEdges[1:] - data.xbinEdges[:-1])
            * np.mean(data.ybinEdges[1:] - data.ybinEdges[:-1])
        )

        xs = np.tile(
            0.5 * (data.xbinEdges[1:] + data.xbinEdges[:-1]),
            (power_density.shape[0], 1),
        )
        ys = np.tile(
            0.5 * (data.ybinEdges[1:] + data.ybinEdges[:-1]),
            (power_density.shape[1], 1),
        ).T

        np.savetxt(
            fname=fname.replace(".npy", ".txt"),
            X=np.stack([xs.flatten(), ys.flatten(), power_density.flatten()], axis=-1),
            delimiter="\t",
            comments="",
            header="""\"X (mm)\"\t\"Y (mm)\"\t\"Power Density (W/mm<sup>2</sup>)\"""",
        )
