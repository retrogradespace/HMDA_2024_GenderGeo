# HMDA 2024: Gender & Geography

## Overview

This repository contains an exploratory analysis of the 2024 Home Mortgage Disclosure Act (HMDA) dataset, focused on gender-based differences in mortgage approval rates and loan pricing across states and metropolitan areas.

## What this project does
- Compares mortgage approval rates by gender across states and metropolitan statistical areas (MSAs).
- Tests for statistically significant gender gaps in rate spreads, with FDR correction across states.
- Fits fixed-effects regressions (state and MSA) estimating the female-applicant rate-spread effect in basis points.
- Produces choropleth maps and heatmaps of approval rates by geography and loan program.

## Tech stack
- Data processing: Pandas, NumPy, SciPy, Statsmodels
- Geospatial: GeoPandas, mapclassify
- Visualization: Seaborn, Matplotlib

## Repository structure
```text
.
├── README.md
├── hmda_loan_analysis_gender_geo_artifactno1.py
├── HMDA_Loan_Analysis_Gender_Geo_ArtifactNo1.ipynb
├── load_hf_dataset.py
├── requirements.txt
├── data/           # Aggregate CSV outputs (approval rates, gap tests, regression results)
├── visuals/        # Choropleth maps, heatmaps, and statistical plots
├── framework/      # Mind map illustrating the analysis framework
└── LICENSE
```

## Getting started

### 1. Create a local environment and install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

### 2. Get the HMDA data
The full raw HMDA file isn't in this repo — it's published separately on Hugging Face at https://huggingface.co/datasets/retrogradespace/hmda_2024. Fetch it and export it to a local CSV:

```bash
python3 load_hf_dataset.py
```

This downloads the dataset and saves it to `~/.cache/hmda_2024/hmda_2024_lar.csv` by default (override with `HMDA_EXPORT_PATH`). Point `HMDA_DATA_PATH` at that file:

```bash
export HMDA_DATA_PATH="$HOME/.cache/hmda_2024/hmda_2024_lar.csv"
```

Keep in mind this is a large ~4.8GB file. A high-memory environment is recommended — if running in Google Colab, use a High-RAM runtime.

### 3. Get the TIGER/Line shapefiles
The state and CBSA geospatial boundaries are separate Census Bureau geography files (not part of the HMDA dataset). Download them from the [Census TIGER/Line page](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html):
- `tl_2024_us_state.zip`
- `tl_2025_us_cbsa.zip`

Point the script at them, or drop them in `data/shapefiles/` (the script's default location):
```bash
export STATE_SHAPEFILE_ZIP="/path/to/tl_2024_us_state.zip"
export CBSA_SHAPEFILE_ZIP="/path/to/tl_2025_us_cbsa.zip"
```

### 4. Run the analysis
```bash
python3 hmda_loan_analysis_gender_geo_artifactno1.py
```

You can also open the notebook in Jupyter for a step-by-step walkthrough (`pip install jupyter` first — it isn't in `requirements.txt`).

## Data
- The full raw HMDA file is published as a separate data release on Hugging Face at https://huggingface.co/datasets/retrogradespace/hmda_2024. See "Get the HMDA data" above for how to fetch it. You can also find it from the source here: https://www.consumerfinance.gov/data-research/hmda/.
- Retrograde Space. (2024). HMDA 2024. Hugging Face Datasets. https://huggingface.co/datasets/retrogradespace/hmda_2024
- TIGER/Line shapefiles are separate Census Bureau geography files: https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html

## Methods
- Winsorization of rate spreads.
- Welch's t-tests for gender gap significance, with FDR correction across states.
- Fixed-effects regression (state and MSA) with cluster-robust standard errors.
- Geospatial mapping using TIGER/Line shapefiles.

## Related work
- [HMDA 2024 Mortgage Profiles](https://github.com/retrogradespace/HMDA_2024_MortgageProfiles): exploratory HMDA data analysis by race and gender, sharing the same published HMDA 2024 dataset on Hugging Face.

## License
This project is distributed under the terms of the included license.
