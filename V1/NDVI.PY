import numpy as np
import rasterio

with rasterio.open("Texas_NDVI_2024.tif") as src:
    ndvi = src.read(1).astype(np.float32)
    nodata = src.nodata

if nodata is not None:
    ndvi[ndvi == nodata] = np.nan
ndvi_real = ndvi / 10000.0
print("Min:", np.nanmin(ndvi))
print("Max:", np.nanmax(ndvi))
print("Mean:", np.nanmean(ndvi))
print("Min:", np.nanmin(ndvi_real))
print("Max:", np.nanmax(ndvi_real))
print("Mean:", np.nanmean(ndvi_real))