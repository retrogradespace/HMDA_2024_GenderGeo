# -*- coding: utf-8 -*-
"""HMDA_Loan_Analysis_Gender_Geo_ArtifactNo1

## Reproduceable Research: Artifact No. 1

In this notebook I perform an exploratory analysis of the `Home Mortgage Disclosure Act` loan application data for 2024 as this was the most recent year that was publicly available. This analysis focuses on understanding approval rates and disparities in rate spreads between female and male applicants across geographical areas.

The National Association of Realtors recently published a report on [Who is Buying Now?](https://www.google.com/url?q=https%3A%2F%2Fwww.nar.realtor%2Fsites%2Fdefault%2Ffiles%2F2025-12%2F2025-12-09-nar-real-estate-forecast-summit-the-year-ahead-jessica-lautz-presentation-slides-who-is-buying-now-12-09-2025.pdf) Single woman made up 25% of first-time homebuyers in 2025 which is up from 11% in 1985. In addition, the growth in purchases by unmarried couples and Other (roomates) also increased. Married couples who in 1985 made up 75% of first-time homebuyers in 2025 only accounted for 50%. The rate of purchases for single-family homes by single-men remained relatively constant.


[Just the Facts, Ma'am: Single Women Home Buyers Since 1981](https://www.nar.realtor/blogs/economists-outlook/just-the-facts-maam-single-women-home-buyers-since-1981)

[2025 Real Estate Forecast: Who is Buying Now](https://www.nar.realtor/sites/default/files/2025-12/2025-12-09-nar-real-estate-forecast-summit-the-year-ahead-jessica-lautz-presentation-slides-who-is-buying-now-12-09-2025.pdf)


**Key Steps Covered:**

- **Data Setup and Loading:** Initial configuration, loading of HMDA data, and preparation of geographical shapefiles.
- **Data Cleaning and Filtering:** Normalizing data codes, applying cohort filters, and flagging key variables like approval status and applicant gender.
- **Statistical Analysis:** Calculating approval rates by state and MSA, performing t-tests to compare rate spreads between genders, and running OLS regression models with fixed effects.
- **Geospatial Visualization:** Creating choropleth maps to visualize approval rates across states and CBSAs.
- **Results Presentation:** Displaying key tables and figures to summarize findings.

**Find the raw data from the source:**

TIGER/Line Shapefiles Format: The core TIGER/Line Files and Shapefiles do not include demographic data, but they do contain geographic entity codes that can be linked to the Census Bureau’s demographic data, available on data.census.gov. https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html

HMDA Dataset: These files contain all data fields available in the public data record and can be used for advanced analysis. There is also an API. For questions/suggestions, contact hmdahelp@cfpb.gov. https://ffiec.cfpb.gov/data-browser/data/2024?category=states

Github Repos:
- [HMDA 2024 Gender & Geography](https://github.com/retrogradespace/HMDA_2024_GenderGeo)
- [HMDA 2024 Mortgage Profiles](https://github.com/retrogradespace/HMDA_2024_MortgageProfiles)

# 2024 HMDA Loan Application Analysis

This analysis of `2024 Home Mortgage Disclosure Act (HMDA)` data shows mortgage approval rates, loan pricing, and geographical patterns for single-party applicants. The global approval rate for single-family, owner-occupied mortgage applicants in the states and territories sampled is just shy of 60%. However, approval rates vary significantly across geographies, with Wisconsin and North Dakota showing the highest rates, while Louisiana and Mississippi have some of the lowest.

A notable finding is a consistent **gender disparity in loan rate spreads**. On average, female applicants receive slightly lower interest rate spreads than male applicants, even after controlling for various loan characteristics and geographical factors (states and metropolitan areas). This "female effect" ranges from **-1.61 to -5.39 basis points lower** depending on the loan program (Conventional, FHA, VA, USDA).

Geographically, this disparity is not uniform. While some states like Mississippi and Connecticut show females receiving significantly lower rates, others such as New York and Illinois demonstrate the opposite, with females paying higher rate spreads. Heatmaps further highlight these complex regional and program-specific variations, indicating distinct patterns in loan accessibility and pricing across the nation.

While these findings are interesting, they are limited to a single year and notably this dataset does not include any information on applicant credit score or income. Finally, I only considered `approved` loans for `residential` occupancy and held fixed `debt_to_income_ratio`, `loan_to_value_ratio`, `reason_for_denial` or `loan_amount` among other variables.

In a related analysis (Artifact No.2) I examine `loan_amount`, `loan_to_value_ratio`, `rate_spread`, and `debt_to_income_ratio` by `derived_sex` and `derived_race`. It appears that single-men apply for a higher average `loan_amount` than single-female applicants, an observation that held true across racial groups. More research would be needed to examine the significance of this relationship. Artifact No. 3 begins the next step of aggregating multi-year data and identifying additional data sources to further this analysis.

### 0) Setup: Mount Drives | Create Output Folders | Build Helper Functions

This section initializes the necessary output directories for figures, tables, and checkpoints. It also defines utility functions for safely saving `pandas` DataFrames to CSV files and `Matplotlib` and `Seaborn`
figures to image files, ensuring that parent directories are created if they don't exist.

Due to the size of this data, a high-memory environment is recommended. If running in Google Colab, select a High-RAM runtime before executing any code (Runtime menu -> Change runtime type).
"""

# 0) Setup: output folders & small helpers
import os
from pathlib import Path

OUT_FIG_DIR = 'output/figures'
OUT_TAB_DIR = 'output/tables'
OUT_CKP_DIR = 'output/checkpoints'

for d in (OUT_FIG_DIR, OUT_TAB_DIR, OUT_CKP_DIR):
    Path(d).mkdir(parents=True, exist_ok=True)

def safe_save_csv(df, path, **kwargs):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, **kwargs)
    print(f"Saved: {p}")

def safe_save_fig(path, dpi=300):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    plt.savefig(p, dpi=dpi)
    print(f"Saved: {p}")

"""### 1) Define Data Paths

Set-up main path to the HMDA data and the geographical shapefiles (for states and CBSAs). Paths are configurable via environment variables so this runs anywhere, not just in the author's original Colab session.

The HMDA source file is published on Hugging Face at https://huggingface.co/datasets/retrogradespace/hmda_2024 -- run `python3 load_hf_dataset.py` first to fetch it and export it locally, then point HMDA_DATA_PATH at the export (see README.md for details).

The state and CBSA TIGER/Line shapefiles are separate Census Bureau geography files (not part of the HMDA dataset) -- download them from https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html and point STATE_SHAPEFILE_ZIP / CBSA_SHAPEFILE_ZIP at the downloaded files.

A quick check to make sure these files are actually there before we start, so we know all our data is ready to go.
"""

# 1) Data paths -- override any of these with environment variables.
CSV_PATH  = os.environ.get('HMDA_DATA_PATH', 'year_2024.csv')
STATE_ZIP = os.environ.get('STATE_SHAPEFILE_ZIP', 'data/shapefiles/tl_2024_us_state.zip')
CBSA_ZIP  = os.environ.get('CBSA_SHAPEFILE_ZIP', 'data/shapefiles/tl_2025_us_cbsa.zip')

assert os.path.exists(CSV_PATH), (
    f"HMDA CSV not found at {CSV_PATH}. Run `python3 load_hf_dataset.py` to fetch it from "
    "Hugging Face, then set HMDA_DATA_PATH to the exported file."
)
assert os.path.exists(STATE_ZIP), f"State shapefile not found: {STATE_ZIP}. Set STATE_SHAPEFILE_ZIP."
assert os.path.exists(CBSA_ZIP), f"CBSA shapefile not found: {CBSA_ZIP}. Set CBSA_SHAPEFILE_ZIP."

print("Paths OK.")

"""### 2) Install `mapclassify` for Choropleth Legends

Intalling the mapclassify library because it helps create clean legends with quantile and fixed break features.
"""

# Commented out IPython magic to ensure Python compatibility.

# 2) (Optional) install mapclassify for quantile/fixed-break legends
# %pip install --quiet mapclassify

"""### 3) Load HMDA Data

Pull HMDA data from CSV file on Google Drive using pandas `read_csv` and generate dataframe of columns for analysis. The `USEGEOCOLS` to restrict is called in the read_csv statment to keep things focused.  In addition, will load as `dtypes=str` and `low_memory=False` because this dataset is pretty big.

Then a peek at what has loaded, by showing the first and last few rows of the DataFrame
"""

# 3) Load HMDA columns
import pandas as pd

USEGEOCOLS = [
    'state_code','derived_msa-md',
    'loan_purpose','occupancy_type','derived_dwelling_category',
    'action_taken','derived_sex',
    'loan_type',
    'rate_spread','loan_amount','income',
    'loan_to_value_ratio','debt_to_income_ratio','interest_rate'
]

df_geo = pd.read_csv(CSV_PATH, usecols=USEGEOCOLS, dtype=str, low_memory=False)
print("Loaded:", df_geo.shape)
print(df_geo.head())
print(df_geo.tail())

"""### 4) Normalize Codes, Rename MSA, Filters & Flags

This next bit cleans up and ensures the data is ready for analysis, so I do a few key things here:
- First, rename that `derived_msa-md` column to `derived_msa_md` it just makes it easier to work with in our code and I hit errors working with this variable in `matplotlib`.
- Some of our categories, like `loan_purpose` and `loan_type`, will be transformed into numbers so they play nicely with the tests we want to run next.
- Then, some filtering since the cohort I run the tests on should have the data we need available for the records included in our study population. In this case, I am interested in residential purchases of single-family homes 1-4 units by single purchasers.
- Add a simple 'yes/no' flags: one for 'approved' loans, and others to easily spot `female_app` and `male_app` in our statistical models.
- A quick conversion for pricing and loan term details turning them into numbers.
- Finally, save all this neatly filtered data as a Parquet file.


"""

# 4) Normalize codes, rename MSA, filters & flags
import numpy as np

to_num = lambda s: pd.to_numeric(s, errors='coerce')

# Safe rename for formulas
if 'derived_msa-md' in df_geo.columns:
    df_geo = df_geo.rename(columns={'derived_msa-md': 'derived_msa_md'})

# Normalize HMDA coded fields
for c in ['loan_purpose','occupancy_type','action_taken','loan_type']:
    df_geo[c] = to_num(df_geo[c])

# Uniform cohort filters
filt = (
    (df_geo['loan_purpose'] == 1) &
    (df_geo['occupancy_type'] == 1) &
    (df_geo['derived_dwelling_category'] == 'Single Family (1-4 Units):Site-Built')
)
df_f = df_geo.loc[filt].copy()

# Flags
df_f['approved']   = (df_f['action_taken'] == 1).astype(int)
df_f['female_app'] = (df_f['derived_sex'] == 'Female')
df_f['male_app']   = (df_f['derived_sex'] == 'Male')

# Pricing/terms
for c in ['rate_spread','loan_amount','income','loan_to_value_ratio','debt_to_income_ratio','interest_rate']:
    df_f[c] = to_num(df_f[c])

print("Cohort rows:", len(df_f))
# Save checkpoint
df_f.to_parquet(f'{OUT_CKP_DIR}/filtered_hmda_2024.parquet', index=False)
print("Saved:", f'{OUT_CKP_DIR}/filtered_hmda_2024.parquet')

"""### 5) Approval Rates & Top/Bottom 10 State with Wilson 95% Confidence Intervals

Figure out the loan approval rates for each state using the Wilson score interval method to get esimate 95% confidence intervals. Helpful when looking at proportions, like approval rates. To make sure the results are reliable, I'm cutting out U.S. territories and any states that don't have a minimum number of applications. Save to CSV then show the Top 10 and Bottom 10 States based on approval rates
"""

# 5) Approval rates by state + Wilson 95% CI
from statsmodels.stats.proportion import proportion_confint

by_state = (
    df_f.groupby('state_code')
        .agg(apps=('approved','size'), approvals=('approved','sum'))
        .assign(approval_rate=lambda d: d['approvals']/d['apps'])
        .reset_index()
)

ci_low, ci_high = proportion_confint(
    by_state['approvals'].astype(int),
    by_state['apps'].astype(int),
    alpha=0.05, method='wilson'
)
by_state['ci_low']  = ci_low
by_state['ci_high'] = ci_high

# Exclude territories (optional) and apply minimum apps
territories = {'PR','GU','VI','AS','MP'}
MIN_APPS = 10_000

robust_rank = (
    by_state[~by_state['state_code'].isin(territories) & (by_state['apps'] >= MIN_APPS)]
    .sort_values('approval_rate', ascending=False)
    .reset_index(drop=True)
)

# Save tables
safe_save_csv(by_state,   f'{OUT_TAB_DIR}/approval_rate_by_state_2024.csv', index=False)
safe_save_csv(robust_rank,f'{OUT_TAB_DIR}/approval_rate_by_state_2024_ci_minapps.csv', index=False)

# Show top/bottom 10 robust ranks
print("Top 10 states (>=10k apps, territories excluded):")
display(robust_rank.head(10))
print("Bottom 10 states (>=10k apps, territories excluded):")
display(robust_rank.tail(10))

print(robust_rank.approval_rate.mean())
print(robust_rank.approval_rate.std())
print(robust_rank.approval_rate.median())
print(robust_rank.approval_rate.min())
print(robust_rank.approval_rate.max())

"""### 6) Female vs. Male Rate Spread Analysis (Winsorized)

Narrow down to just the approved loans with good rate spread data. Then, ensure extreme values don't throw off our results, will `winsorize` the rate spread data – capping the highest and lowest 1% of values. After that, run a two-sample t-test (a Welch's t-test, to be precise) to see if there's a statistically significant difference in the average rate spreads for female versus male applicants. Generate and save some basic descriptive stats for export and print
"""

# 6) Female vs Male (winsorized) on approved loans
import scipy.stats as stats

cond = df_f[(df_f['approved']==1) & (df_f['rate_spread'].notna()) & (df_f['derived_sex'].isin(['Female','Male']))].copy()

q1, q99 = cond['rate_spread'].quantile([0.01, 0.99])
cond_ws = cond.assign(rate_spread=cond['rate_spread'].clip(lower=q1, upper=q99))

female_rs = cond_ws.loc[cond_ws['derived_sex']=='Female','rate_spread'].dropna()
male_rs   = cond_ws.loc[cond_ws['derived_sex']=='Male','rate_spread'].dropna()

diff  = float(female_rs.mean() - male_rs.mean())
tstat, pval = stats.ttest_ind(female_rs, male_rs, equal_var=False)

print(f"Unadjusted mean diff (Female - Male), winsorized: {diff:.4f}  (~{diff*100:.2f} bps)")
print(f"t-stat = {tstat:.2f}, p = {pval:.4g}")

desc = cond_ws.groupby('derived_sex')['rate_spread'].describe()
display(desc)

# Save tables
safe_save_csv(desc,      f'{OUT_TAB_DIR}/rate_spread_descriptive_by_sex_approved_2024.csv')
safe_save_csv(cond_ws[['derived_sex','rate_spread']], f'{OUT_TAB_DIR}/rate_spread_winsorized_pairs_2024.csv', index=False)

"""### 7) Adjusted OLS Fixed Effects Models with Cluster-Robust Standard Errors
Use adjusted OLS regression, to further example `female effect` on rate spreads. Control for loan details and account for differences across states and metro areas.

Build two main models:
1.  **State-level Model:** This one looks at the big picture, accounting for unique characteristics of each state.
2.  **Metro-level Model:** This dives a bit deeper, considering the specific dynamics of different metropolitan areas (MSAs).


"""

# 7) Adjusted OLS FE + cluster-robust SEs (replacement)
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd

# --- Build regression sample (approved + rate_spread present; already winsorized in cond_ws) ---
reg = cond_ws.dropna(subset=['rate_spread','loan_amount','income','loan_to_value_ratio','debt_to_income_ratio']).copy()
reg['loan_amount_k'] = reg['loan_amount'] / 1000.0  # scaling for stability

# --- Helpers: build aligned model frames for State FE and MSA FE ---
def make_state_df(reg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with NA for all variables appearing in the State-FE formula.
    Create cluster labels aligned to model rows.
    """
    needed = [
        'rate_spread','female_app','loan_amount_k','income',
        'loan_to_value_ratio','debt_to_income_ratio','state_code','loan_type'
    ]
    mdf = reg_df.dropna(subset=needed).copy()
    # Clean cluster labels (numeric codes for statsmodels)
    mdf['cluster_state'] = pd.Categorical(mdf['state_code'].astype(str).fillna('UNKNOWN')).codes
    return mdf

def make_msa_df(reg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with NA for all variables appearing in the MSA-FE formula.
    Create cluster labels aligned to model rows.
    """
    needed = [
        'rate_spread','female_app','loan_amount_k','income',
        'loan_to_value_ratio','debt_to_income_ratio','derived_msa_md','loan_type'
    ]
    mdf = reg_df.dropna(subset=needed).copy()
    mdf['cluster_msa'] = pd.Categorical(mdf['derived_msa_md'].astype(str).fillna('UNKNOWN')).codes
    return mdf

state_df = make_state_df(reg)
msa_df   = make_msa_df(reg)

print("STATE model rows:", len(state_df), "| clusters:", len(np.unique(state_df['cluster_state'])))
print("MSA   model rows:", len(msa_df),   "| clusters:", len(np.unique(msa_df['cluster_msa'])))

# --- Fit models (State FE and MSA FE), clustering on geography ---
model_state = smf.ols(
    'rate_spread ~ female_app + loan_amount_k + income + loan_to_value_ratio + debt_to_income_ratio + C(state_code) + C(loan_type)',
    data=state_df
).fit(cov_type='cluster', cov_kwds={'groups': state_df['cluster_state']})

model_msa = smf.ols(
    'rate_spread ~ female_app + loan_amount_k + income + loan_to_value_ratio + debt_to_income_ratio + C(derived_msa_md) + C(loan_type)',
    data=msa_df
).fit(cov_type='cluster', cov_kwds={'groups': msa_df['cluster_msa']})

print("=== STATE FE (clustered by state) ===")
print(model_state.summary())
print("\n=== MSA FE (clustered by MSA) ===")
print(model_msa.summary())

# --- Robust extractor: handles 'female_app' or 'female_app[T.True]' ---
def summarize_female_effect_any(results, label, base='female_app'):
    """
    Return coef & 95% CI in basis points for either 'female_app' or 'female_app[T.True]'.
    """
    candidates = [base, f'{base}[T.True]']
    present = [name for name in candidates if name in results.params.index]
    if not present:
        return {'model': label, 'coef_bps': np.nan, 'ci_low_bps': np.nan, 'ci_high_bps': np.nan}
    name = present[0]

    coef = results.params[name]
    ci   = results.conf_int().loc[name] if name in results.conf_int().index else [np.nan, np.nan]

    return {
        'model': label,
        'coef_bps': float(coef) * 100.0,
        'ci_low_bps': float(ci[0]) * 100.0,
        'ci_high_bps': float(ci[1]) * 100.0
    }

# --- Build and save OVERALL FE summary (in bps) ---
summ_df_fixed = pd.DataFrame([
    summarize_female_effect_any(model_state, 'State FE + loan_type'),
    summarize_female_effect_any(model_msa,   'MSA FE + loan_type')
])
display(summ_df_fixed)

summ_df_fixed.to_csv(f'{OUT_TAB_DIR}/female_effect_bps_overall_FE_2024.csv', index=False)
print("Saved:", f'{OUT_TAB_DIR}/female_effect_bps_overall_FE_2024.csv')

# 7a. Program-stratified FE summaries (rebuild & save)
program_rows = []
programs = {1:'Conventional', 2:'FHA', 3:'VA', 4:'USDA'}

for code, label in programs.items():
    reg_lt_state = state_df[state_df['loan_type'] == code]
    reg_lt_msa   = msa_df[msa_df['loan_type']   == code]

    # State FE within program
    if len(reg_lt_state) >= 5000:
        mod_state_lt = smf.ols(
            'rate_spread ~ female_app + loan_amount_k + income + loan_to_value_ratio + debt_to_income_ratio + C(state_code)',
            data=reg_lt_state
        ).fit(cov_type='cluster', cov_kwds={'groups': reg_lt_state['cluster_state']})
        program_rows.append(summarize_female_effect_any(mod_state_lt, f'{label} — State FE'))

    # MSA FE within program
    if len(reg_lt_msa) >= 5000:
        mod_msa_lt = smf.ols(
            'rate_spread ~ female_app + loan_amount_k + income + loan_to_value_ratio + debt_to_income_ratio + C(derived_msa_md)',
            data=reg_lt_msa
        ).fit(cov_type='cluster', cov_kwds={'groups': reg_lt_msa['cluster_msa']})
        program_rows.append(summarize_female_effect_any(mod_msa_lt, f'{label} — MSA FE'))

prog_summ_fixed = pd.DataFrame(program_rows)
display(prog_summ_fixed)

prog_summ_fixed.to_csv(f'{OUT_TAB_DIR}/female_effect_bps_program_FE_2024.csv', index=False)
print("Saved:", f'{OUT_TAB_DIR}/female_effect_bps_program_FE_2024.csv')

import pandas as pd

print(pd.read_csv(f'{OUT_TAB_DIR}/female_effect_bps_overall_FE_2024.csv'))
print(pd.read_csv(f'{OUT_TAB_DIR}/female_effect_bps_program_FE_2024.csv').head())

"""### 8) Program-Stratified Fixed Effects Models

**Diving Deeper: How Loan Programs Influence the `Female Effect`**
See if women face different rate spreads in, say, a Conventional loan compared to an FHA loan, even after controlling for various other factors.
"""

# 8) Program-stratified FE models (replacement)
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd

# Robust extractor: handles 'female_app' or 'female_app[T.True]'
def summarize_female_effect_any(results, label, base='female_app'):
    candidates = [base, f'{base}[T.True]']
    present = [name for name in candidates if name in results.params.index]
    if not present:
        return {'model': label, 'coef_bps': np.nan, 'ci_low_bps': np.nan, 'ci_high_bps': np.nan}
    name = present[0]
    coef = results.params[name]
    ci = results.conf_int().loc[name] if name in results.conf_int().index else [np.nan, np.nan]
    return {
        'model': label,
        'coef_bps': float(coef) * 100.0,
        'ci_low_bps': float(ci[0]) * 100.0,
        'ci_high_bps': float(ci[1]) * 100.0
    }

# Helper: build program-specific model frame with aligned cluster labels
def make_program_state_df(state_df_full: pd.DataFrame, lt_code: int) -> pd.DataFrame:
    needed = ['rate_spread','female_app','loan_amount_k','income',
              'loan_to_value_ratio','debt_to_income_ratio','state_code']
    mdf = state_df_full[state_df_full['loan_type'] == lt_code].dropna(subset=needed).copy()
    # Rebuild cluster codes on the filtered frame
    mdf['cluster_state'] = pd.Categorical(mdf['state_code'].astype(str).fillna('UNKNOWN')).codes
    return mdf

def make_program_msa_df(msa_df_full: pd.DataFrame, lt_code: int) -> pd.DataFrame:
    needed = ['rate_spread','female_app','loan_amount_k','income',
              'loan_to_value_ratio','debt_to_income_ratio','derived_msa_md']
    mdf = msa_df_full[msa_df_full['loan_type'] == lt_code].dropna(subset=needed).copy()
    mdf['cluster_msa'] = pd.Categorical(mdf['derived_msa_md'].astype(str).fillna('UNKNOWN')).codes
    return mdf

programs = {1:'Conventional', 2:'FHA', 3:'VA', 4:'USDA'}
rows = []

for code, label in programs.items():
    # Build aligned frames per program
    reg_lt_state = make_program_state_df(state_df, code)
    reg_lt_msa   = make_program_msa_df(msa_df,   code)

    # Fit State FE within program (skip small samples)
    if len(reg_lt_state) >= 5000:
        try:
            mod_state_lt = smf.ols(
                'rate_spread ~ female_app + loan_amount_k + income + loan_to_value_ratio + debt_to_income_ratio + C(state_code)',
                data=reg_lt_state
            ).fit(cov_type='cluster', cov_kwds={'groups': reg_lt_state['cluster_state']})
            rows.append(summarize_female_effect_any(mod_state_lt, f'{label} — State FE'))
        except Exception as e:
            print(f"Warning: {label} — State FE failed: {e}")
    else:
        print(f"Skipping {label} — State FE (sample < 5,000): n={len(reg_lt_state)}")

    # Fit MSA FE within program (skip small samples)
    if len(reg_lt_msa) >= 5000:
        try:
            mod_msa_lt = smf.ols(
                'rate_spread ~ female_app + loan_amount_k + income + loan_to_value_ratio + debt_to_income_ratio + C(derived_msa_md)',
                data=reg_lt_msa
            ).fit(cov_type='cluster', cov_kwds={'groups': reg_lt_msa['cluster_msa']})
            rows.append(summarize_female_effect_any(mod_msa_lt, f'{label} — MSA FE'))
        except Exception as e:
            print(f"Warning: {label} — MSA FE failed: {e}")
    else:
        print(f"Skipping {label} — MSA FE (sample < 5,000): n={len(reg_lt_msa)}")

# Build summary DataFrame & save
prog_summ = pd.DataFrame(rows)
display(prog_summ)
safe_save_csv(prog_summ, f'{OUT_TAB_DIR}/female_effect_bps_program_FE_2024.csv', index=False)
print("Saved:", f'{OUT_TAB_DIR}/female_effect_bps_program_FE_2024.csv')

# Visualize program effects with 95% CI bars
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = prog_summ.copy()
df['program'] = df['model'].str.split(' — ').str[0]
df['fe_type'] = df['model'].str.split(' — ').str[1]

# Order programs
order = ['Conventional', 'FHA', 'VA', 'USDA']
df['program'] = pd.Categorical(df['program'], categories=order, ordered=True)

# Plot
plt.figure(figsize=(9,5))
for i, fe in enumerate(['State FE', 'MSA FE']):
    sub = df[df['fe_type'] == fe].sort_values('program')
    x = np.arange(len(sub))
    y = sub['coef_bps'].values
    yerr = np.vstack([y - sub['ci_low_bps'].values, sub['ci_high_bps'].values - y])
    plt.errorbar(x + (i-0.5)*0.15, y, yerr=yerr, fmt='o', capsize=4,
                 label=fe, color=('#1f77b4' if fe=='State FE' else '#ff7f0e'))

plt.axhline(0, color='gray', linewidth=1)
plt.xticks(np.arange(len(order)), order)
plt.ylabel('Female − Male rate spread (bps)')
plt.title('Adjusted female effect by program (95% CI)')
plt.legend()
plt.tight_layout()
plt.savefig(f'{OUT_FIG_DIR}/female_effect_by_program_bars_2024.png', dpi=300)
plt.show()

print("Saved:", f'{OUT_FIG_DIR}/female_effect_by_program_bars_2024.png')

"""### 9) State-Level Female vs. Male Rate Spread Differences with FDR Correction

I was curious if loan pricing varies significantly for women and men, even after accounting for various factors. For each state where there is enough data from both female and male applicants, perform a **Welch's t-test**. Try to pinpoint if there are truly significant differences in the *average rate spread* between these two groups.

Running many tests (one for each state!), there's a risk of finding "significant" results purely by chance. To address this, apply a **False Discovery Rate (FDR) correction** to p-values. This ensures our findings are more robust and less likely to be false positives.

Finally, present these results, including a clear `significance flag` to highlight where these differences are most pronounced.
"""

# 9) State-level Female vs Male differences (+ FDR)
from scipy.stats import ttest_ind
import numpy as np

rows = []
for st, g in cond_ws.groupby('state_code'):
    f = g[g['derived_sex']=='Female']['rate_spread'].dropna()
    m = g[g['derived_sex']=='Male']['rate_spread'].dropna()
    if len(f) > 100 and len(m) > 100:
        t, p = ttest_ind(f, m, equal_var=False)
        rows.append({'state_code': st, 'n_female': len(f), 'n_male': len(m),
                     'diff_mean_bps': (f.mean()-m.mean())*100.0, 't': t, 'p': p})

state_tests = pd.DataFrame(rows).sort_values('diff_mean_bps')
state_tests['rank']   = np.arange(1, len(state_tests)+1)
state_tests['p_fdr']  = state_tests['p'] * len(state_tests) / state_tests['rank']
state_tests['sig_fdr_5pct'] = state_tests['p_fdr'] <= 0.05

display(state_tests.head(10))
display(state_tests.tail(10))
safe_save_csv(state_tests, f'{OUT_TAB_DIR}/state_rate_spread_female_male_tests_FDR_2024.csv', index=False)

"""### 10) CBSA Aggregates & Merge TIGER CBSA Shapefile

This section takes the loan approval rates and groups them by metropolitan statistical area (CBSA) using the `derived_msa_md` column. Then, pull in a geographical shapefile for CBSAs and unzipping it, then joining to the calculated approval rates.
"""

# 10) CBSA aggregates & merge TIGER CBSA
import zipfile, glob, os
import geopandas as gpd

# Aggregate approval rates by HMDA metro code
by_msa = (
    df_f[df_f['derived_msa_md'].notna()]
    .groupby('derived_msa_md')
    .agg(apps=('approved','size'), approvals=('approved','sum'))
    .assign(approval_rate=lambda d: d['approvals']/d['apps'])
    .reset_index()
)
by_msa['derived_msa_md'] = by_msa['derived_msa_md'].astype(str)
safe_save_csv(by_msa, f'{OUT_TAB_DIR}/approval_rate_by_msa_2024.csv', index=False)

# Unzip CBSA shapefile
CBSA_DIR = os.path.splitext(CBSA_ZIP)[0] + '/'
os.makedirs(CBSA_DIR, exist_ok=True)
with zipfile.ZipFile(CBSA_ZIP, 'r') as z:
    z.extractall(CBSA_DIR)

cbsa_shp = glob.glob(os.path.join(CBSA_DIR, '*.shp'))[0]
gdf_cbsa = gpd.read_file(cbsa_shp)

# Join key: CBSAFP (string) ←→ derived_msa_md
gdf_cbsa['CBSAFP'] = gdf_cbsa['CBSAFP'].astype(str)
gdf_cbsa_merged = gdf_cbsa.merge(
    by_msa.rename(columns={'derived_msa_md':'CBSAFP'}),
    on='CBSAFP', how='left'
)
print("CBSA merged:", gdf_cbsa_merged.shape)

"""### 11) Unzip State Shapefile and Merge with State-Level Data

State-level geographical data. Unzip the TIGER/Line shapefile for U.S. states and then combine it with previously calculated state-level approval rates (from the `by_state` DataFrame). Use `STUSPS` and `state_code` to make sure everything links up correctly. Once merged, this GeoDataFrame is all set for creating state-level choropleth maps.
"""

# 11) Unzip state shapefile, merge with by_state
STATE_DIR = os.path.splitext(STATE_ZIP)[0] + '/'
os.makedirs(STATE_DIR, exist_ok=True)
with zipfile.ZipFile(STATE_ZIP, 'r') as z:
    z.extractall(STATE_DIR)

state_shp = glob.glob(os.path.join(STATE_DIR, '*.shp'))[0]
gdf_states = gpd.read_file(state_shp)

# Common join key: STUSPS -> state_code
gdf_states = gdf_states.rename(columns={'STUSPS':'state_code'})
gdf_state_merged = gdf_states.merge(by_state, on='state_code', how='left')
print("State merged:", gdf_state_merged.shape)

"""### 12) Helper Function: Choropleth with Quantile/Fixed-Break Legends and Labels

This cell defines a super handy Python function called `plot_choropleth_with_labels`. It's designed to create beautiful choropleth maps.

This function is really versatile, supporting different ways to classify our data (like Quantiles or custom User-Defined fixed breaks) and allowing use of various color schemes.

Plus, it can add clear labels to the map.

I commented out the labels (which you can remove the `#` to try out), I just found it too busy on the figures.
"""

# 12) Helper: choropleth with quantile/fixed breaks + labels
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

def plot_choropleth_with_labels(
    gdf, value_col, title, outfile,
    scheme='Quantiles', bins=None, k=5, cmap='Blues',
    edgecolor='0.8', linewidth=0.5,
    legend_loc='lower left', legend_title=None,
  #  label_col='NAME', label_top_n=20, label_fontsize=9,
  #  use_halo=True, halo_color='white', halo_width=2
):
    import mapclassify  # ensure available
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1,1, figsize=(14,8))
    class_kwargs = {}
    if scheme == 'UserDefined':
        assert bins and isinstance(bins, (list, tuple)), "Provide bins for UserDefined."
        class_kwargs = {'bins': bins}

    gdf.plot(
        column=value_col,
        scheme=scheme, k=(k if scheme=='Quantiles' else None),
        classification_kwds=(class_kwargs if scheme=='UserDefined' else None),
        cmap=cmap, linewidth=linewidth, edgecolor=edgecolor,
        legend=True, ax=ax,
        missing_kwds={'color':'lightgrey','label':'No data'},
        legend_kwds={'loc': legend_loc, 'title': legend_title or value_col}
    )

   # to_label = (
    #    gdf.dropna(subset=[value_col, label_col])
     #      .sort_values(value_col, ascending=False)
      #     .head(label_top_n)
   # )
   # rep_points = to_label.representative_point()
   # pathfx = [pe.withStroke(linewidth=halo_width, foreground=halo_color)] if use_halo else []

   # for (x, y), txt in zip(rep_points.geometry.apply(lambda p: (p.x, p.y)), to_label[label_col]):
    #    ax.annotate(txt, xy=(x, y), ha='center', va='center',
     #               fontsize=label_fontsize, fontweight='bold', color='#1f2937',
      #              path_effects=pathfx)

    ax.set_title(title, fontsize=14)
    ax.set_axis_off()
    plt.tight_layout()
    safe_save_fig(outfile, dpi=300)
    plt.show()

"""### 13) Render Choropleth Maps

This section calls the `plot_choropleth_with_labels` helper function to generate four different choropleth maps.

- State-level Quantile Map: This one shows approval rates by state, dividing them into equal groups based on the number of observations
- State-level Fixed Breaks Map: visualizing state approval rates using custom, pre-defined ranges.
- CBSA-level Quantile Map: Approval rates for metropolitan areas, again divided by quantiles.
- CBSA-level Fixed Breaks Map: And finally, metropolitan area approval rates but with hard-coded fixed breaks.

Each of these maps will be saved as a PNG image in the output/figures directory, so they can easily share and review them. I decided to get rid of all labels because at this zoom it makes this just way to busy.

Note to self: Figure out how to plot the labels off center or nest a heat map near by.

"""

# 13) Render maps
# State — Quantiles + abbreviations
plot_choropleth_with_labels(
    gdf=gdf_state_merged, value_col='approval_rate',
    title='Approval Rates by State (2024 HMDA filtered cohort)',
    outfile=f'{OUT_FIG_DIR}/choropleth_state_quantiles_abbrev_2024.png',
    scheme='Quantiles', k=5, cmap='Blues',
    legend_title='Approval rate (quantiles)',
    #label_col='state_code', label_top_n=50, label_fontsize=9
)

# State — Fixed breaks + abbreviations
fixed_bins_state = [0.60, 0.70, 0.80, 0.90]  # tune to distribution
plot_choropleth_with_labels(
    gdf=gdf_state_merged, value_col='approval_rate',
    title='Approval Rates by State (2024; fixed breaks)',
    outfile=f'{OUT_FIG_DIR}/choropleth_state_fixed_abbrev_2024.png',
    scheme='UserDefined', bins=fixed_bins_state, cmap='Blues',
    legend_title='Approval rate (fixed breaks)',
    #label_col='state_code', label_top_n=50, label_fontsize=9
)

# CBSA — Quantiles + limited labels
plot_choropleth_with_labels(
    gdf=gdf_cbsa_merged, value_col='approval_rate',
    title='Approval Rates by Metro (CBSA) — 2024 HMDA filtered cohort',
    outfile=f'{OUT_FIG_DIR}/choropleth_cbsa_quantiles_labels_2024.png',
    scheme='Quantiles', k=6, cmap='Purples',
    legend_title='Approval rate (quantiles)',
    #label_col='NAME', label_top_n=30, label_fontsize=7.5
)

# CBSA — Fixed breaks + limited labels
fixed_bins_cbsa = [0.55, 0.65, 0.75, 0.85]  # tune to distribution
plot_choropleth_with_labels(
    gdf=gdf_cbsa_merged, value_col='approval_rate',
    title='Approval Rates by Metro (CBSA) — 2024; fixed breaks',
    outfile=f'{OUT_FIG_DIR}/choropleth_cbsa_fixed_labels_2024.png',
    scheme='UserDefined', bins=fixed_bins_cbsa, cmap='Purples',
    legend_title='Approval rate (fixed breaks)',
    #label_col='NAME', label_top_n=30, label_fontsize=7.5
)

"""### 14) Bar Charts with 95% Confidence Intervals for Top/Bottom States


**Top 10 States:**
This chart highlights the approval rates for the top 10 states (those meeting our minimum application threshold and excluding territories). Includes their 95% Wilson confidence intervals.

**Bottom 10 States:**
Similarly, this chart shows the approval rates for the bottom 10 states (again, meeting our inclusion criteria), and their 95% Wilson confidence intervals.

These charts make it super easy to visually grasp the variation in state-level approval rates and understand how reliable our estimates are. Both figures will be saved as handy PNG images.

Note to self: Any improvement would be to plot these on the same figure come back and revise.
"""

# 14) Bar charts with 95% CIs for top/bottom states
import numpy as np
import matplotlib.pyplot as plt

top10 = robust_rank.head(10)
plt.figure(figsize=(10,6))
plt.errorbar(
    x=np.arange(len(top10)),
    y=top10['approval_rate'],
    yerr=[top10['approval_rate'] - top10['ci_low'], top10['ci_high'] - top10['approval_rate']],
    fmt='o', ecolor='gray', capsize=4, color='#1f2937'
)
plt.xticks(np.arange(len(top10)), top10['state_code'], rotation=0)
plt.ylim(0, 1)
plt.ylabel('Approval rate')
plt.title('Top 10 Approval Rates (95% CI; >=10k apps; territories excluded)')
plt.tight_layout()
safe_save_fig(f'{OUT_FIG_DIR}/bar_top10_approval_rate_ci.png', dpi=300)
plt.show()

bottom10 = robust_rank.tail(10)
plt.figure(figsize=(10,6))
plt.errorbar(
    x=np.arange(len(bottom10)),
    y=bottom10['approval_rate'],
    yerr=[bottom10['approval_rate'] - bottom10['ci_low'], bottom10['ci_high'] - bottom10['approval_rate']],
    fmt='o', ecolor='gray', capsize=4, color='#1f2937'
)
plt.xticks(np.arange(len(bottom10)), bottom10['state_code'], rotation=0)
plt.ylim(0, 1)
plt.ylabel('Approval rate')
plt.title('Bottom 10 Approval Rates (95% CI; >=10k apps; territories excluded)')
plt.tight_layout()
safe_save_fig(f'{OUT_FIG_DIR}/bar_bottom10_approval_rate_ci.png', dpi=300)
plt.show()

"""### 15) Sanity Check

Time for a quick check-up! This section performs a 'sanity check' on the data and the environment. Print out the versions of all the key Python libraries we're using (like `NumPy`, `pandas`, `GeoPandas`, `Shapely`, and `mapclassify`).

Quick summary statistics on our filtered HMDA cohort, including the total number of rows, applications, and approvals, plus the overall approval rate. Display the counts of approved applications that have valid rate spread data, broken down by applicant gender.

Just a little peek behind the curtain to ensure everything's looking good!
"""

# 15) Sanity Check
import numpy as np, pandas as pd
try:
    import geopandas as gpd, shapely, mapclassify
    print("NumPy:", np.__version__, "| pandas:", pd.__version__,
          "| GeoPandas:", gpd.__version__, "| Shapely:", shapely.__version__,
          "| mapclassify:", mapclassify.__version__)
except Exception as e:
    print("Version check:", e)

# Cohort sanity
total_rows = len(df_f)
apps = len(df_f)
approvals = int(df_f['approved'].sum())
print(f"Cohort rows: {total_rows} | Applications: {apps} | Approvals: {approvals} "
      f"({approvals/apps:.1%})")

# Approved + valid rate_spread + sex
mask = (df_f['approved']==1) & (df_f['rate_spread'].notna())
counts = df_f.loc[mask, 'derived_sex'].value_counts(dropna=False)
print("Approved & rate_spread available by sex:\n", counts)

"""### 16) Print Results for Top/Bottom States

This cell simply re-displays the top and bottom 10 states by approval rate. Calculated in Section 5, but it's handy to see them again here for a quick overview of which states are leading and lagging in approval rates, after filtering out territories and those with too few applications.
"""

# 16) print results
print("Top 10 states (>=10k apps, territories excluded):")
display(robust_rank.head(10))
print("Bottom 10 states (>=10k apps, territories excluded):")
display(robust_rank.tail(10))

"""### 17) Rate Spread Analysis by Sex (Pre-Winsorization)

How available and distributed is the `rate_spread` information for approved loans when broken down by applicant sex. First, calculate the frequency of `rate_spread` data and count of records available for `female`, `male`, `joint`, and `sex not available` applications.

Then, look at the percentiles (specifically the 1st, 25th, 50th, 75th, and 99th) of the `rate_spread` for `female` and `male` applicants. Helps to understand the spread of the data, especially the extreme ends, before applying any winsorization. Raw view of the distribution tails.
"""

# 17) Rate Spread by sex
# Availability of rate_spread among approvals by sex
mask = (df_f['approved']==1)
avail = (
    df_f.loc[mask]
        .assign(has_rs = df_f.loc[mask,'rate_spread'].notna())
        .groupby('derived_sex')['has_rs']
        .agg(['sum','count'])
        .assign(availability=lambda d: d['sum']/d['count'])
)
display(avail)

# Tails by sex (pre-winsorization)
for s in ['Female','Male']:
    series = df_f.loc[mask & (df_f['derived_sex']==s),'rate_spread'].dropna()
    q = series.quantile([0.01,0.25,0.50,0.75,0.99])
    print(f"\n{ s } rate_spread quantiles:\n{ q }")

"""### 18) Suggest Fixed Breaks for Choropleth Map

This cell sets 'fixed breaks' for the choropleth maps, which are calculated based on the percentiles of the approval rates from our state ranking. Calculate the 10th, 30th, 50th, 70th, and 90th percentiles of the `approval_rate` for states that have enough data.
"""

# 18) Suggest fixed breaks from percentiles of the robust set
import numpy as np

vals = robust_rank['approval_rate'].dropna().values
p = np.percentile(vals, [10, 30, 50, 70, 90])  # edit percentiles to taste
bins_suggested = [float(x) for x in p]
print("Suggested bins (UserDefined):", bins_suggested)

"""### 19) Choropleth Map of State Approval Rates with Tuned Fixed Breaks

This section creates a choropleth map that visualizes state-level approval rates using the custom set of fixed breaks. This is different from a quantile-based map because it manually tweaks the boundaries of the classes to highlight specific ranges of approval rates.

To update these fixed breaks return to Step 18 and update the breaks that are hard coded in this statement:

`p = np.percentile(vals, [10, 30, 50, 70, 90])`

Originally, the mapping included state abbreviations as labels but it made the visual way to busy so they are commented out currently.

Remove the `#` from these two lines before run to show labels

`#label_col='state_code', label_top_n=50, label_fontsize=9,`
`#use_halo=True, halo_color='white', halo_width=2`

Then save the final image to output directory.
"""

# 19) Fixed breaks tuned to your distribution (five classes)
fixed_bins_state = [0.52, 0.56, 0.60, 0.64, 0.68]

plot_choropleth_with_labels(
    gdf=gdf_state_merged, value_col='approval_rate',
    title='Approval Rates by State (2024; fixed breaks)',
    outfile=f'{OUT_FIG_DIR}/choropleth_state_fixed_abbrev_2024.png',
    scheme='UserDefined', bins=fixed_bins_state, cmap='Blues',
    legend_title='Approval rate (fixed breaks)',
   # label_col='state_code', label_top_n=50, label_fontsize=9,  # abbreviations
   # use_halo=True, halo_color='white', halo_width=2
)

"""### 20) Bar Chart: Top 10 State Approval Rates with Confidence Intervals

Generate an error bar chart that displays the approval rates for the top 10 states (pulled from `robust_rank` DataFrame) and include the 95% Wilson confidence intervals.

This chart gives a visual representation of which states have the highest approval rates and how precise those estimates are.


"""

# 20) top 10 approval
import numpy as np
import matplotlib.pyplot as plt

top10 = robust_rank.head(10)
plt.figure(figsize=(10,6))
plt.errorbar(
    x=np.arange(len(top10)), y=top10['approval_rate'],
    yerr=[top10['approval_rate'] - top10['ci_low'], top10['ci_high'] - top10['approval_rate']],
    fmt='o', ecolor='gray', capsize=4, color='#1f2937'
)
plt.xticks(np.arange(len(top10)), top10['state_code'], rotation=0)
plt.ylim(0.5, 0.75)
plt.ylabel('Approval rate')
plt.title('Top 10 Approval Rates (95% CI; ≥10k apps; territories excluded)')
plt.tight_layout()
safe_save_fig(f'{OUT_FIG_DIR}/bar_top10_approval_rate_ci.png', dpi=300)
plt.show()

"""### 21) Save Descriptive Statistics and Winsorized Rate Spread Pairs

This section stores data outputs to CSV file:

**rate_spread_descriptive_by_sex_approved_2024.csv**: This file contains all the descriptive statistics (like count, mean, standard deviation, minimum, quartiles, and maximum) for the winsorized rate spread data. Grouped by applicant sex for approved loans.

**rate_spread_winsorized_pairs_2024.csv**: This file holds the derived_sex and rate_spread columns specifically for approved and winsorized loans. This is the exact data we used in our t-test and OLS regression analysis, so it's perfect for anyone wanting to dive deeper or reproduce our results.
"""

# 22) Save descriptive stats & the winsorized pairs used for the t-test
safe_save_csv(desc, f'{OUT_TAB_DIR}/rate_spread_descriptive_by_sex_approved_2024.csv')
safe_save_csv(cond_ws[['derived_sex','rate_spread']], f'{OUT_TAB_DIR}/rate_spread_winsorized_pairs_2024.csv', index=False)

"""### 22) Save Unadjusted Female-Male Rate Spread Gap (Winsorized)

Package the unadjusted mean difference in winsorized rate spreads between female and male applicants for approved loans.

Create a `DataFrame` that includes everything: the female mean, the male mean, the gap expressed in basis points, the t-statistic, Welch's degrees of freedom, and the 95% confidence interval for that gap.

This `DataFrame` is then saved as **unadjusted_female_male_gap_winsorized_2024.csv**. Clear summary of the direct comparison of rate spreads by gender before any statistical adjustments are applied.
"""

# 22) Plug values computed should have written to variables inline maybe go back and revise later
import pandas as pd, math

# Plug in the values computed
gap_bps       = -1.419192674270714
se_pct_points = 0.001983596925647966
t_stat        = -7.154642437284064
df_welch      = 1235588.6738683356
ci_low_bps    = -1.8079776716977154
ci_high_bps   = -1.0304076768437125

unadj = pd.DataFrame([{
    'cohort': 'Approved + valid rate_spread (winsorized 1%/99%)',
    'female_mean': 0.07030325776479059,
    'male_mean':   0.08449518450749773,
    'gap_bps':     gap_bps,
    't_stat':      t_stat,
    'df_welch':    df_welch,
    'ci95_low_bps': ci_low_bps,
    'ci95_high_bps': ci_high_bps
}])

unadj.to_csv('output/tables/unadjusted_female_male_gap_winsorized_2024.csv', index=False)
print("Saved:", 'output/tables/unadjusted_female_male_gap_winsorized_2024.csv')

"""### 23) Build CBSA-By-State Approval Tables

Random selection of state approval tables on a few key states (like Texas, Florida, Georgia, Virginia, and Vermont – easily tweak that list for different set of states!). For each of the named states, aggregate the approval rates by CBSA (metropolitan statistical area) and then sort them to see which metros are leading. Save these detailed, state-specific CBSA approval rate tables as CSV files, will allow exploration of the nuances in each state's metropolitan lending landscape.
"""

# 23) Build CBSA-by-state approval tables (e.g., for TX, FL, GA, VA)
states_focus = ['TX','FL','GA','VA', 'VT', 'WI', 'ND', 'NH', 'CT', 'SD', 'AR', 'LA']  # edit if you want others
rows = []
for st in states_focus:
    g = df_f[(df_f['state_code']==st) & (df_f['derived_msa_md'].notna())].copy()
    by_cbsa_st = (
        g.groupby('derived_msa_md')
         .agg(apps=('approved','size'), approvals=('approved','sum'))
         .assign(approval_rate=lambda d: d['approvals']/d['apps'])
         .reset_index()
         .sort_values('approval_rate', ascending=False)
    )
    by_cbsa_st.to_csv(f'output/tables/cbsa_approval_{st}_2024.csv', index=False)
    print("Saved:", f'output/tables/cbsa_approval_{st}_2024.csv')

"""### 24) Rate Spread Distribution by Gender (Winsorized)

This cell plots the distribution of the `Winsorized` rate spread data for approved loans, colored by applicant gender. Using a Kernel Density Estimate (KDE) plot, to see the shape and distributions for male and female applicants.

This visual does indicate a difference in loan packages.

*Note to self: Go back and create a distribution plot before normalizing as well, just because, maybe later.*
"""

# 24) Rate Distro by derived_sex

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming cond_ws DataFrame from your pipeline
plt.figure(figsize=(8,5))
sns.kdeplot(data=cond_ws, x="rate_spread", hue="derived_sex", fill=True, common_norm=False)
plt.title("Distribution of Rate Spread by Gender (Winsorized)")
plt.xlabel("Rate Spread")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig('output/figures/rate_spread_distribution_winsorized.png', dpi=300)
plt.show()
print("Saved: rate_spread_distribution_winsorized.png")

"""### 25) Approval Rate Heatmap by State and Loan Program

Generate a heatmap displaying the average loan approval rates across U.S. states and loan programs (Conventional, FHA, VA, USDA).

Provide a high-level visual overview of how approval rates vary geographically and by the type of loan product offered. The color intensity indicates the approval rate for each state-loan program combination.

It was the first attempt and I needed to spend sometime in the documentation. There is a lot of improvement left to do, but this is a start. https://matplotlib.org/stable/users/explain/quick_start.html#coding-styles
"""

# 25 Approval Rate Heatmap by State and Loan Program
import seaborn as sns
import matplotlib.pyplot as plt

# Pivot table for heatmap
pivot = df_f.groupby(['state_code','loan_type'])['approved'].mean().unstack()

# Plot heatmap with adjustments for labels
plt.figure(figsize=(14, 18))  # Increase height for all states
sns.heatmap(pivot, cmap="Blues", annot=True, fmt=".2f", cbar_kws={'label': 'Approval Rate'})

# Title and axis labels
plt.title("Approval Rate by State and Loan Program", fontsize=16)
plt.xlabel("Loan Type", fontsize=14)
plt.ylabel("State Code", fontsize=14)

# Ensure all state labels are visible
plt.xticks(rotation=0, fontsize=12)
plt.yticks(rotation=0, fontsize=10)  # Keep states horizontal for readability

# Save to PNG
plt.tight_layout()
plt.savefig('output/figures/approval_rate_heatmap_state_program.png', dpi=300)

plt.show()
print("Saved: approval_rate_heatmap_state_program.png")

"""### 26) Diverging Approval Rate Heatmap by State and Loan Program

Building upon the previous heatmap, this cell provides a more refined visualization of approval rates by state and loan program. It maps numeric loan type codes to descriptive names and uses a diverging color scheme centered at 0.6. This allows for clearer identification of states and loan programs with approval rates significantly above or below this central threshold. Thin black lines are added to enhance visual separation between cells.

*Note to self: figure out how to get rid of the vertical lines, really want just the horizontal ones.*
"""

#26 Formatted with different color ramp: https://matplotlib.org/stable/users/explain/colors/colormaps.html
import seaborn as sns
import matplotlib.pyplot as plt

# Map loan type codes to names
loan_type_map = {1: 'Conventional', 2: 'FHA', 3: 'VA', 4: 'USDA'}

# Pivot table and rename columns
pivot = df_f.groupby(['state_code','loan_type'])['approved'].mean().unstack()
pivot.rename(columns=loan_type_map, inplace=True)

# Plot heatmap with diverging colors and boundaries
plt.figure(figsize=(14, 18))
sns.heatmap(
    pivot,
    cmap="RdBu_r",
    center=0.6,
    annot=True,
    fmt=".2f",
    cbar_kws={'label': 'Approval Rate'},
    linecolor='black',
    linewidths=0.5
)

plt.title("Approval Rate by State and Loan Program", fontsize=16)
plt.xlabel("Loan Program", fontsize=14)
plt.ylabel("State Code", fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(rotation=0, fontsize=10)

plt.tight_layout()
plt.savefig('output/figures/approval_rate_heatmap_state_program_diverging.png', dpi=300)
plt.show()
print("Saved: approval_rate_heatmap_state_program_diverging.png")

"""### 28) Approval Rate Heatmap by MSA and Loan Program

This cell generates a heatmap to visualize loan approval rates across Metropolitan Statistical Areas (MSAs) and different loan programs. To ensure readability, it first loads CBSA shapefile data to create a lookup table, mapping the numerical MSA IDs (`derived_msa_md`) to their corresponding geographical names. This allows the heatmap to display meaningful metro names on the y-axis, providing insights into metro-specific approval rate patterns.
"""

# 28) Approval rate heatmap with properly formated columns

import geopandas as gpd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load CBSA shapefile for metro names
cbsa_shapefile = glob.glob(os.path.join(CBSA_DIR, '*.shp'))[0]
gdf_cbsa = gpd.read_file(cbsa_shapefile)

# 2. Create lookup table for CBSA ID → NAME
cbsa_lookup = gdf_cbsa[['CBSAFP', 'NAME']].rename(columns={'CBSAFP': 'derived_msa_md'})
cbsa_lookup['derived_msa_md'] = cbsa_lookup['derived_msa_md'].astype(str)

# 3. Pivot HMDA data by metro and loan type
pivot_msa = df_f.groupby(['derived_msa_md', 'loan_type'])['approved'].mean().unstack()
loan_type_map = {1: 'Conventional', 2: 'FHA', 3: 'VA', 4: 'USDA'}
pivot_msa.rename(columns=loan_type_map, inplace=True)

# Replace index IDs with metro names
pivot_msa.index = pivot_msa.index.map(cbsa_lookup.set_index('derived_msa_md')['NAME'])

# ✅ Drop rows where index is NaN (unmapped metro IDs)
pivot_msa = pivot_msa[~pivot_msa.index.isna()]


# 4. Plot optimized heatmap https://matplotlib.org/stable/users/explain/colors/colormaps.html
plt.figure(figsize=(16, len(pivot_msa)*0.35))  # Dynamic height
sns.heatmap(
    pivot_msa,
    cmap="seismic",          # Diverging color ramp
    center=0.6,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 7, "weight": "bold"},
    cbar_kws={'label': 'Approval Rate'},
    linecolor='black',        # Horizontal grid lines only
    linewidths=0.5
)

# Remove vertical grid lines
for spine in plt.gca().spines.values():
    spine.set_visible(False)

# Titles and labels
plt.title("Approval Rate by MSA (Metro Name) and Loan Program", fontsize=20, fontweight='bold')
plt.xlabel("Loan Program", fontsize=14, fontweight='bold')
plt.ylabel("Metro Area", fontsize=14, fontweight='bold')
plt.xticks(rotation=45, fontsize=13, fontweight='bold')
plt.yticks(rotation=0, fontsize=8, fontweight='bold')

# Save and show
plt.tight_layout()
plt.savefig('output/figures/approval_rate_heatmap_msa_program_final.png', dpi=400)
plt.show()
print("Saved: approval_rate_heatmap_msa_program_final.png")

"""### 29) Sorted Approval Rate Heatmap by MSA and Loan Program

This enhanced heatmap presents loan approval rates by MSA and loan program, with the MSAs sorted by their Conventional loan approval rates in descending order. This sorting strategy helps to quickly identify and compare MSAs that excel or underperform in Conventional loan approvals, and observe how this performance correlates across other loan programs. Dynamic sizing and optimized annotations are used for improved clarity.
"""

# 29) Heatmap for MSA sorted by conventional loan approvals descending
import geopandas as gpd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load CBSA shapefile for metro names
cbsa_shapefile = glob.glob(os.path.join(CBSA_DIR, '*.shp'))[0]
gdf_cbsa = gpd.read_file(cbsa_shapefile)

# 2. Create lookup table for CBSA ID → NAME
cbsa_lookup = gdf_cbsa[['CBSAFP', 'NAME']].rename(columns={'CBSAFP': 'derived_msa_md'})
cbsa_lookup['derived_msa_md'] = cbsa_lookup['derived_msa_md'].astype(str)

# 3. Pivot HMDA data by metro and loan type
pivot_msa = df_f.groupby(['derived_msa_md', 'loan_type'])['approved'].mean().unstack()
loan_type_map = {1: 'Conventional', 2: 'FHA', 3: 'VA', 4: 'USDA'}
pivot_msa.rename(columns=loan_type_map, inplace=True)

# Replace index IDs with metro names
pivot_msa.index = pivot_msa.index.map(cbsa_lookup.set_index('derived_msa_md')['NAME'])

# ✅ Drop rows where index is NaN (unmapped metro IDs)
pivot_msa = pivot_msa[~pivot_msa.index.isna()]


# Sort by overall approval rate (mean across loan types)
pivot_msa['avg_rate'] = pivot_msa.mean(axis=1)

# Sort by Conventional loan approval rate
pivot_msa = pivot_msa.sort_values('Conventional', ascending=False)

# 4. Plot optimized heatmap https://matplotlib.org/stable/users/explain/colors/colormaps.html
plt.figure(figsize=(16, len(pivot_msa)*0.35))  # Dynamic height
sns.heatmap(
    pivot_msa,
    cmap="seismic",          # Diverging color ramp
    center=0.6,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 7, "weight": "bold"},
    cbar_kws={'label': 'Approval Rate'},
    linecolor='black',        # Horizontal grid lines only
    linewidths=0.5
)

# Remove vertical grid lines
for spine in plt.gca().spines.values():
    spine.set_visible(False)

# Titles and labels
plt.title("Approval Rate by MSA (Metro Name) and Loan Program", fontsize=20, fontweight='bold')
plt.xlabel("Loan Program", fontsize=14, fontweight='bold')
plt.ylabel("Metro Area", fontsize=14, fontweight='bold')
plt.xticks(rotation=45, fontsize=13, fontweight='bold')
plt.yticks(rotation=0, fontsize=8, fontweight='bold')

# Save and show
plt.tight_layout()
plt.savefig('output/figures/approval_rate_heatmap_msa_program_final.png', dpi=400)
plt.show()
print("Saved: approval_rate_heatmap_msa_program_final.png")

"""### 30) Sorted Approval Rate Heatmap by State and Loan Program

This cell visualizes loan approval rates across states and loan programs, with the states sorted by their Conventional loan approval rates in descending order. This sorting helps in identifying states that have higher (or lower) approval rates for Conventional loans and allows for a quick comparison of their performance across other loan programs (FHA, VA, USDA).
"""

# 30) Re-Sort of State Heat map on Approval for conventional loans descending
import seaborn as sns
import matplotlib.pyplot as plt

# Map loan type codes to names
loan_type_map = {1: 'Conventional', 2: 'FHA', 3: 'VA', 4: 'USDA'}

# Pivot table and rename columns
pivot = df_f.groupby(['state_code','loan_type'])['approved'].mean().unstack()
pivot.rename(columns=loan_type_map, inplace=True)

pivot = pivot.sort_values('Conventional', ascending=False)

# Plot heatmap with diverging colors and boundaries
plt.figure(figsize=(14, 18))
sns.heatmap(
    pivot,
    cmap="seismic",
    center=0.6,
    annot=True,
    fmt=".2f",
    cbar_kws={'label': 'Approval Rate'},
    linecolor='black',
    linewidths=0.5
)

plt.title("Approval Rate by State and Loan Program", fontsize=16)
plt.xlabel("Loan Program", fontsize=14)
plt.ylabel("State Code", fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(rotation=0, fontsize=10)

plt.tight_layout()
plt.savefig('output/figures/approval_rate_heatmap_state_program_diverging.png', dpi=300)
plt.show()
print("Saved: approval_rate_heatmap_state_program_diverging.png")