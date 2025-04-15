import pickle
import numpy as np
from PIL import Image
import os, sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Pass the .npy filename as an argument")
        sys.exit(1)

    fname = sys.argv[1]
    if not os.path.exists(fname):
        print("File not found")
        sys.exit(1)

    with open(fname, "rb") as f:
        f = pickle.load(f)
        tmp = f.total2D.copy()
        tmp = ((tmp / tmp.max()) * int("1" * 16, 2)).astype(np.uint16)
        img = Image.fromarray(tmp)
        f2name = ".".join(fname.split(".") + ["tiff"])
        img.save(f2name)
        print(f'16-bit TIFF "{f2name}" written successfully')
