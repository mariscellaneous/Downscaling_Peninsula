
---

# MERRA-2 Neural Network Downscaling (1 km; 3-Hourly) — Antarctic Peninsula

Spatial downscaling of atmospheric reanalysis 2-m air temperature data (t2m) over the Antarctic Peninsula using a digital elevation model (1 km).

These are the following steps:

1. Loads a grid from a MATLAB file (topography, land/ocean mask, indexing, etc.).
2. Reads **MERRA-2** 3-hourly surface variables from NetCDF files.
3. Trains a small **PyTorch neural network** to learn relationships between grid-based predictors (lat/lon/elevation/hour + neighbor fields) and MERRA-2 **T2M**.
4. Applies the trained model to a higher-resolution/polar stereographic grid (e.g., ice surface/topography points) via interpolation and reconstruction.
5. Extracts predicted temperatures at **AWS station locations**, compares them to raw MERRA-2 and to station observations, and saves plots + `.mat` outputs.


---

![](Figures/sample_output.png)

---

## What this script produces

* **Downscaled temperatures at station points** (`results`) at 3-hourly resolution
* **Original MERRA-2 temperatures at station points** (`weknows`) at 3-hourly resolution

---

## Installation

Recommended: use a clean Python virtual environment (e.g., `venv`).

### Python dependencies

Core scientific stack:

* `numpy`
* `scipy`
* `pandas`
* `matplotlib`
* `scikit-learn`
* `netCDF4`

ML:

* `torch`
* `tqdm`

Mapping/projection:

* `cartopy`

---

## Key parameters

* Geographic filtering is done using a mask like:

  * Latitude range: `[-76, -61]`
  * Longitude range: `(-80.625, -47.5]`
  * Excludes ocean (`frocean < 1`)
  * Excludes NaNs

---

**Inputs (20 features total):**

* elevation
* hour-of-day
* 8 neighbor elevations (rolled)
* 8 neighbor temperatures (rolled)

**Target:**

* MERRA-2 `T2M`

**Architecture:**

* Linear(20 → 64) + ReLU
* Linear(64 → 64) + ReLU
* Linear(64 → 1)

**Training:**

* Loss: MSE
* Optimizer: Adam (`lr=1e-4`)
* Train/test split: 80/20
* Epochs: 100
* StandardScaler normalization on features

---

## Citation

MERRA-2: Gelaro, R., McCarty, W., Suárez, M. J., Todling, R., Molod, A., Takacs, L., ... & Zhao, B. (2017). The modern-era retrospective analysis for research and applications, version 2 (MERRA-2). Journal of climate, 30(14), 5419-5454.

REMA (DEM): Howat, I. M., Porter, C., Smith, B. E., Noh, M. J., & Morin, P. (2019). The reference elevation model of Antarctica. The Cryosphere, 13(2), 665-674.
