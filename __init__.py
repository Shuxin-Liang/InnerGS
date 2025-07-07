import sys
sys.path.insert(0, r"C:\Users\uamls\Desktop\2dgs-cuda")

import importlib.util
spec = importlib.util.find_spec("raster.diff_gaussian_rasterization._C")
print(spec)
