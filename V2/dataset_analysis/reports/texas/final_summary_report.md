# Wildfire Dataset — Complete Exploratory Analysis Report

> **Generated:** 2026-07-03 10:01:51  
> **Dataset:** Historical Wildfire Records 2014–2020 (FPA-FOD)  
> **Pipeline:** Wildfire Prediction System — Dataset Analysis Module  

---

## Table of Contents
1. [Dataset Overview](#1-dataset-overview)
2. [Schema & Feature Summary](#2-schema--feature-summary)
3. [Feature Categories](#3-feature-category-distribution)
4. [Missing Value Summary](#4-missing-value-summary)
5. [Feature Quality Summary](#5-feature-quality-summary)
6. [Statistical Summary](#6-statistical-summary)
7. [Temporal Analysis](#7-temporal-analysis)
8. [Geographic Analysis](#8-geographic-analysis)
9. [Correlation Summary](#9-correlation-summary)
10. [Categorical Analysis](#10-categorical-analysis)
11. [Leakage Analysis](#11-leakage-analysis)
12. [Predictive Readiness](#12-predictive-readiness)
13. [Feature Dependency Groups](#13-feature-dependency-groups)
14. [Source Readiness](#14-source-readiness-update-frequency)
15. [Recommended Next Steps](#15-recommended-next-steps)

---


## 1. Dataset Overview

| Property | Value |
|----------|-------|
| **State** | Texas |
| **Total Rows** | 51,033 |
| **Total Columns** | 309 |
| **Memory Usage (MB)** | 278.77 MB |
| **Dtype Breakdown** | float64: 251; object: 54; int64: 4 |
| **Numeric Columns** | 255 |
| **Categorical Columns** | 30 |
| **Datetime Columns** | 0 |
| **Boolean Columns** | 0 |
| **Object Columns (raw)** | 54 |
| **Text Columns** | 24 |
| **Float64 Columns** | 251 |
| **Int64 Columns** | 4 |
| **Year Range** | 2014 – 2020 |
| **Data Source** | FPA-FOD Pre-processed (Phase-1 Pipeline) |



## 2. Schema & Feature Summary

- Total features analyzed: **309**

### Semantic Type Breakdown
- **Numeric**: 255 columns
- **Categorical**: 30 columns
- **Text**: 24 columns

### Sample — First 20 Features
| Column Name | Data Type | Semantic Type | Unique Values | Missing % | Description |
| --- | --- | --- | --- | --- | --- |
| FOD_ID | int64 | Numeric | 51033 | 0.0 | Fire occurrence database unique identifier |
| FPA_ID | object | Text | 51033 | 0.0 | No description available |
| SOURCE_SYSTEM_TYPE | object | Categorical | 3 | 0.0 | No description available |
| SOURCE_SYSTEM | object | Categorical | 6 | 0.0 | No description available |
| NWCG_REPORTING_AGENCY | object | Categorical | 6 | 0.0 | Agency that reported the fire |
| NWCG_REPORTING_UNIT_ID | object | Categorical | 33 | 0.0 | No description available |
| NWCG_REPORTING_UNIT_NAME | object | Categorical | 33 | 0.0 | No description available |
| SOURCE_REPORTING_UNIT | object | Categorical | 39 | 0.0 | No description available |
| SOURCE_REPORTING_UNIT_NAME | object | Categorical | 46 | 0.0 | No description available |
| LOCAL_FIRE_REPORT_ID | float64 | Numeric | 284 | 99.1202 | No description available |
| LOCAL_INCIDENT_ID | object | Text | 50072 | 1.3031 | No description available |
| FIRE_CODE | object | Text | 614 | 98.4912 | No description available |
| FIRE_NAME | object | Text | 37488 | 0.143 | Name assigned to the fire |
| ICS_209_PLUS_INCIDENT_JOIN_ID | object | Text | 442 | 99.1241 | Join key to ICS-209+ incident database |
| ICS_209_PLUS_COMPLEX_JOIN_ID | object | Text | 10 | 99.9667 | Join key to ICS-209+ complex database |
| MTBS_ID | object | Text | 159 | 99.6826 | Monitoring Trends in Burn Severity fire ID |
| MTBS_FIRE_NAME | object | Text | 157 | 99.6826 | Fire name in MTBS dataset |
| COMPLEX_NAME | object | Text | 18 | 99.9491 | Name of fire complex (if part of multi-fire event) |
| FIRE_YEAR | int64 | Numeric | 7 | 0.0 | Calendar year the fire was discovered |
| DISCOVERY_DATE | object | Text | 2345 | 0.0 | Date fire was first reported |

_Showing top 20 of 309 rows._



## 3. Feature Category Distribution

| Category | Column Count |
| --- | --- |
| Other | 181 |
| Fire Weather | 32 |
| Administrative | 19 |
| Fire Metadata | 15 |
| Terrain | 10 |
| Geographic | 8 |
| Weather Normal | 8 |
| Vegetation Index | 6 |
| Infrastructure | 6 |
| Fire Outcome | 5 |
| Vegetation | 5 |
| Fire Stations | 4 |
| Weather | 4 |
| Cause | 2 |
| Population | 2 |
| Temporal | 1 |
| Climate | 1 |



## 4. Missing Value Summary

| Metric | Value |
|--------|-------|
| Total Columns | 309 |
| Fully Complete | N/A |
| Has Any Missing | 195 |

### Missing Tier Breakdown
| Tier | Column Count |
| --- | --- |
| Complete (0%) | 114 |
| Critical (≥90.0%) | 23 |
| High (≥75.0%) | 2 |
| Low (≥25.0%) | 6 |
| Minimal (<25%) | 164 |


### Critical (≥90.0% missing)
  - `GACC_Fire Use Teams`
  - `IALMIL_87`
  - `geometry`
  - `IAPLHS_88`
  - `IAULHS_89`
  - `IAHSEF`
  - `ICS_209_PLUS_COMPLEX_JOIN_ID`
  - `COMPLEX_NAME`
  - `road_interstate_dis`
  - `road_US_dis`
  - `MTBS_ID`
  - `MTBS_FIRE_NAME`
  - `road_other_dis`
  - `NWCG_CAUSE_AGE_CATEGORY`
  - `ICS_209_PLUS_INCIDENT_JOIN_ID`
  - `LOCAL_FIRE_REPORT_ID`
  - `road_state_dis`
  - `Evacuation`
  - `FIRE_CODE`
  - `road_county_dis`

### High (≥75.0% missing)
  - `CONT_DATE`
  - `CONT_DOY`

### Moderate (≥50.0% missing)
  _None_

> **Note:** No values were imputed. No columns were removed. This is a read-only analysis.



## 5. Feature Quality Summary

| Metric | Value |
|--------|-------|
| Total Columns | 309 |
| PASS (no issues) | N/A |
| REVIEW (has issues) | 88 |

- **Constant columns**: `NWCG_CAUSE_AGE_CATEGORY`, `STATE`, `GACC_Fire Use Teams`, `IA_LMI_ET`, `IA_UN_ET`, `IA_POV_ET`, `IAULHSE`, `IAPLHSE`, `IALMILHSE`, `IALMIL_87`, `IAPLHS_88`, `IAULHS_89`, `IALHE`, `IAHSEF`, `UI_EXP`, `THRHLD`, `tmmn_Percentile`, `tmmx_Percentile`, `geometry`
- **Duplicate columns**: `FIRE_YEAR`, `GACC_Fire Use Teams`, `M_WTR`, `SM_C`, `SM_PFS`, `WDLI`, `WD_ET`, `IA_LMI_ET`, `IA_UN_ET`, `IA_POV_ET`, `IAULHSE`, `IAPLHSE`, `IALMILHSE`, `IALMIL_87`, `IAPLHS_88`, `IAULHS_89`, `LHE`, `IALHE`, `IAHSEF`, `M_WTR_EOMI`
- **Columns with infinite values**: _None_



## 6. Statistical Summary

- Numeric columns analyzed: **255**
- Highly skewed columns (|skew| > 2): **111**

### Top 15 by Absolute Skewness
| Column | Skewness | Mean | Median | Std | Outlier % |
| --- | --- | --- | --- | --- | --- |
| SDI | -225.904847 | -6.667888359268098e+31 | 0.05 | 1.5063083011174964e+34 | 0.4174 |
| HWLI | 225.900421 | 2e-05 | 0.0 | 0.004427 | 0.002 |
| FRG | -207.586803 | 2.26526 | 2.0 | 45.527822 | 0.7368 |
| FIRE_SIZE | 155.645016 | 43.03084 | 1.0 | 1645.502225 | 11.7473 |
| TSDF_ET | 75.282431 | 0.000176 | 0.0 | 0.013279 | 0.0176 |
| EPL_MINRTY | -51.781079 | 0.122847 | 0.4704 | 19.286217 | 0.0372 |
| EPL_CROWD | -51.779388 | 0.241209 | 0.6533 | 19.288712 | 0.0372 |
| EPL_MUNIT | -51.779068 | -0.124813 | 0.2472 | 19.281686 | 0.0372 |
| EPL_LIMENG | -51.777442 | 0.179977 | 0.576 | 19.287771 | 0.0372 |
| EPL_SNGPNT | -51.777253 | 0.07178 | 0.4154 | 19.285706 | 0.0372 |
| EPL_AGE17 | -51.776796 | 0.196368 | 0.5884 | 19.288168 | 0.0372 |
| EPL_AGE65 | -51.776269 | 0.243921 | 0.67 | 19.289151 | 0.0372 |
| EPL_GROUPQ | -51.760725 | 0.018203 | 0.3889 | 19.286723 | 0.0372 |
| EPL_NOHSDP | -50.469897 | 0.266819 | 0.681 | 19.7901 | 0.0392 |
| EPL_UNEMP | -50.463759 | 0.02256 | 0.3946 | 19.786064 | 0.0392 |



## 7. Temporal Analysis

### Fires by Year
| Year | Fire Count | % of Total | Total Acres | Mean Acres | Median Acres |
| --- | --- | --- | --- | --- | --- |
| 2014.0 | 8538.0 | 16.73 | 143843.03 | 16.847391660810494 | 1.0 |
| 2015.0 | 8304.0 | 16.27 | 208761.73 | 25.13990004816956 | 1.0 |
| 2016.0 | 8586.0 | 16.82 | 306421.94 | 35.6885557884929 | 1.0 |
| 2017.0 | 8922.0 | 17.48 | 752278.1 | 84.3172046626317 | 1.0 |
| 2018.0 | 7533.0 | 14.76 | 475705.73 | 63.14957254745785 | 1.0 |
| 2019.0 | 6455.0 | 12.65 | 206158.01 | 31.93772424477149 | 1.0 |
| 2020.0 | 2695.0 | 5.28 | 102824.34 | 38.1537439703154 | 1.0 |


### Fires by Month
| Month | Fire Count | Month Name | % of Total |
| --- | --- | --- | --- |
| 1 | 5582 | Jan | 10.94 |
| 2 | 4817 | Feb | 9.44 |
| 3 | 4727 | Mar | 9.26 |
| 4 | 3024 | Apr | 5.93 |
| 5 | 3428 | May | 6.72 |
| 6 | 4104 | Jun | 8.04 |
| 7 | 6526 | Jul | 12.79 |
| 8 | 6015 | Aug | 11.79 |
| 9 | 4102 | Sep | 8.04 |
| 10 | 4032 | Oct | 7.9 |
| 11 | 2476 | Nov | 4.85 |
| 12 | 2200 | Dec | 4.31 |


### Duration Statistics
| Mean Duration (days) | Median Duration (days) | Max Duration (days) | P90 Duration (days) | P95 Duration (days) | Fires with Duration | Fires Missing Duration |
| --- | --- | --- | --- | --- | --- | --- |
| 0.37 | 0.0 | 39.0 | 1.0 | 2.0 | 5149.0 | 45884.0 |



## 8. Geographic Analysis

### Top States by Fire Count
| State | Fire Count | % of Total |
| --- | --- | --- |
| TX | 51033 | 100.0 |


### Top 15 Counties
| County | Fire Count | % of Total |
| --- | --- | --- |
| Upshur | 1438 | 2.8178 |
| McLennan | 1354 | 2.6532 |
| Tarrant | 1353 | 2.6512 |
| Hill | 1323 | 2.5924 |
| Dallas | 1166 | 2.2848 |
| Rusk | 1033 | 2.0242 |
| Cherokee | 931 | 1.8243 |
| Taylor | 814 | 1.595 |
| Navarro | 740 | 1.45 |
| Erath | 700 | 1.3717 |
| Victoria | 697 | 1.3658 |
| Coryell | 689 | 1.3501 |
| Grimes | 661 | 1.2952 |
| Hunt | 651 | 1.2756 |
| Starr | 645 | 1.2639 |


### Top Ecoregions
_No data available._



## 9. Correlation Summary

### Top 20 Highly Correlated Pairs (Pearson)
| Feature A | Feature B | Correlation |
| --- | --- | --- |
| FIRE_YEAR | source_year | 1.0 |
| M_WTR | WDLI | 1.0 |
| LHE | M_WKFC_105 | 1.0 |
| WD_ET | M_WTR_EOMI | 1.0 |
| SM_C | SM_PFS | 1.0 |
| Medusahead | PoaSecunda | 0.999997 |
| RPL_THEMES | RPL_THEME1 | 0.999992 |
| RPL_THEMES | RPL_THEME4 | 0.999982 |
| CheatGrass | ExoticAnnualGrass | 0.99998 |
| CheatGrass | Medusahead | 0.999969 |
| CheatGrass | PoaSecunda | 0.999967 |
| EPL_POV | EPL_NOVEH | 0.999964 |
| RPL_THEME1 | RPL_THEME4 | 0.999963 |
| EPL_PCI | RPL_THEME3 | 0.99995 |
| EPL_MINRTY | EPL_LIMENG | 0.999948 |
| EPL_SNGPNT | EPL_MINRTY | 0.999933 |
| EPL_MINRTY | EPL_CROWD | 0.999922 |
| ExoticAnnualGrass | Medusahead | 0.999921 |
| DISCOVERY_DOY | CONT_DOY | 0.999919 |
| EPL_POV | EPL_MOBILE | 0.999919 |


### Top 20 Highly Correlated Pairs (Spearman)
| Feature A | Feature B | Correlation |
| --- | --- | --- |
| LHE | M_WKFC_105 | 1.0 |
| LOCAL_FIRE_REPORT_ID | road_state_dis | 1.0 |
| FIRE_YEAR | source_year | 1.0 |
| DISCOVERY_TIME | road_interstate_dis | -1.0 |
| SM_C | SM_PFS | 1.0 |
| WD_ET | M_WTR_EOMI | 1.0 |
| M_WTR | WDLI | 1.0 |
| DISCOVERY_DOY | CONT_DOY | 0.999876 |
| CA | NCA | -0.999866 |
| Elevation_1km | Elevation | 0.998684 |
| LILHSE | LISO_ET | 0.998349 |
| pr_5D_mean | pr_5D_max | 0.998026 |
| TC | CC | 0.995899 |
| fm1000 | fm1000_5D_mean | 0.995771 |
| fm1000_5D_mean | fm1000_5D_min | 0.99556 |
| EVH | EVC | 0.992553 |
| Annual_precipitation | Aridity_index | 0.992298 |
| fm1000 | fm1000_5D_min | 0.991801 |
| TRI_1km | Slope_1km | 0.990512 |
| fm1000_Normal | erc_Normal | -0.988712 |



## 10. Categorical Analysis

| Column | Unique Count | Missing % | Top Category | Top Category Count | Top Category % | Top 5 Categories |
| --- | --- | --- | --- | --- | --- | --- |
| FPA_ID | 51033 | 0.0 | SFO-2014TXTXS1873 | 1 | 0.002 | SFO-2014TXTXS1873: 1; SFO-2017TXPRI691301: 1; SFO-2017TXPRI692443: 1; SFO-2017TXPRI691571: 1; SFO-2017TXPRI693578: 1 |
| LOCAL_INCIDENT_ID | 50072 | 1.3031 | 03 | 9 | 0.0176 | 03: 9; 01: 7; 001: 7; 2: 6; 02: 5 |
| EVT_1km | 47130 | 0.0 | 7997(53%) / 7519(35%) / 9823(4%) | 36 | 0.0705 | 7997(53%) / 7519(35%) / 9823(4%): 36; 7997(43%) / 9322(21%) / 7371(11%): 21; 7987(68%) / 7984(16%) / 7299(5%): 17; 7997(28%) / 7299(14%) / 7371(12%): 16; 7987(78%) / 7986(5%) / 9323(5%): 14 |
| Land_Cover_1km | 46587 | 0.0 | 52(100%) | 48 | 0.0941 | 52(100%): 48; 81(53%) / 43(33%) / 71(5%): 36; 81(44%) / 42(28%) / 90(12%): 21; 81(79%) / 90(7%) / 21(5%): 17; 81(30%) / 42(21%) / 43(20%): 16 |
| EVH_1km | 44470 | 0.0 | 304(35%) / 305(13%) / 303(9%) | 38 | 0.0745 | 304(35%) / 305(13%) / 303(9%): 38; 305(21%) / 303(13%) / 304(9%): 21; 304(45%) / 64(16%) / 303(15%): 17; 304(15%) / 25(14%) / 305(11%): 17; 304(45%) / 303(41%) / 25(2%): 14 |
| MOD_EVI_12m | 41236 | 0.0 | '0.21' '0.31' '0.19' '0.24' '0.23' '0.28' '0.29' '0.32' '0.33' '0.33' '0.32' '0.26' | 16 | 0.0314 | '0.21' '0.31' '0.19' '0.24' '0.23' '0.28' '0.29' '0.32' '0.33' '0.33' '0.32' '0.26': 16; '0.55' '0.54' '0.58' '0.51' '0.32' '0.29' '0.3' '0.31' '0.32' '0.39' '0.47' '0.51': 15; '0.26' '0.27' '0.3' '0.27' '0.22' '0.15' '0.14' '0.17' '0.19' '0.2' '0.2' '0.22': 15; '0.24' '0.21' '0.21' '0.23' '0.27' '0.32' '0.33' '0.31' '0.35' '0.36' '0.32' '0.29': 13; '0.29' '0.35' '0.33' '0.23' '0.18' '0.15' '0.17' '0.21' '0.26' '0.3' '0.33' '0.4': 12 |
| MOD_NDVI_12m | 41236 | 0.0 | '0.36' '0.46' '0.34' '0.39' '0.38' '0.42' '0.43' '0.45' '0.46' '0.46' '0.46' '0.41' | 16 | 0.0314 | '0.36' '0.46' '0.34' '0.39' '0.38' '0.42' '0.43' '0.45' '0.46' '0.46' '0.46' '0.41': 16; '0.8' '0.81' '0.84' '0.79' '0.66' '0.67' '0.67' '0.7' '0.74' '0.81' '0.81' '0.8': 15; '0.37' '0.39' '0.45' '0.4' '0.35' '0.27' '0.25' '0.31' '0.32' '0.3' '0.33' '0.34': 15; '0.4' '0.36' '0.35' '0.39' '0.43' '0.46' '0.48' '0.49' '0.47' '0.49' '0.45' '0.43': 13; '0.44' '0.5' '0.48' '0.38' '0.32' '0.29' '0.34' '0.39' '0.42' '0.45' '0.5' '0.54': 12 |
| NDVI_max | 41218 | 0.0 | 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' | 18 | 0.0353 | 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan': 18; '0.38' '0.33' '0.34' '0.41' '0.44' '0.45' '0.48' '0.48' '0.5' '0.54' '0.51' '0.44': 16; '0.85' '0.79' '0.83' '0.85' '0.73' '0.7' '0.7' '0.7' '0.79' '0.84' '0.89' '0.81': 15; '0.42' '0.46' '0.48' '0.44' '0.32' '0.28' '0.26' '0.28' '0.37' '0.37' '0.38' '0.43': 15; '0.46' '0.37' '0.38' '0.38' '0.45' '0.5' '0.51' '0.52' '0.55' '0.59' '0.52' '0.5': 13 |
| NDVI_mean | 41218 | 0.0 | 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' | 18 | 0.0353 | 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan': 18; '0.22' '0.2' '0.12' '0.2' '0.27' '0.29' '0.26' '0.33' '0.34' '0.24' '0.25' '0.26': 16; '0.62' '0.4' '0.28' '0.29' '0.31' '0.31' '0.37' '0.17' '0.41' '0.59' '0.37' '0.47': 15; '0.31' '0.29' '0.17' '0.17' '0.13' '0.16' '0.15' '0.12' '0.2' '0.27' '0.18' '0.27': 15; '0.17' '0.16' '0.21' '0.15' '0.24' '0.39' '0.31' '0.34' '0.32' '0.31' '0.29' '0.29': 13 |
| NDVI_min | 41160 | 0.0 | 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' | 18 | 0.0353 | 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan': 18; '0.01' '0.01' '-0.0' '-0.03' '0.01' '0.01' '0.0' '0.06' '0.0' '0.0' '-0.01' '-0.0': 16; '0.04' '-0.0' '-0.0' '0.0' '-0.01' '-0.0' '-0.0' '-0.04' '-0.0' '0.03' '0.0' '-0.01': 15; '-0.01' '-0.02' '-0.0' '-0.0' '-0.0' '0.0' '-0.01' '-0.02' '0.01' '0.03' '-0.0' '-0.01': 15; '-0.01' '-0.0' '-0.01' '0.0' '-0.02' '0.02' '0.01' '0.05' '0.01' '-0.01' '-0.01' '0.02': 13 |
| EVC_1km | 41117 | 0.0 | 355(10%) / 162(4%) / 161(3%) | 36 | 0.0705 | 355(10%) / 162(4%) / 161(3%): 36; 355(14%) / 171(5%) / 170(4%): 21; 355(9%) / 25(6%) / 364(5%): 18; 355(11%) / 25(7%) / 22(4%): 18; 25(14%) / 355(8%) / 22(4%): 17 |
| FIRE_NAME | 37488 | 0.143 | GRASS FIRE | 3515 | 6.8877 | GRASS FIRE: 3515; BRUSH FIRE: 279; UNKNOWN: 157; GRASS FIRE : 145; GRASS: 142 |
| FRG_1km | 11435 | 0.0 | 2(100%) | 3253 | 6.3743 | 2(100%): 3253; 1(100%): 1160; 2(100%) / 1(0%): 893; 1(100%) / 111(0%): 577; 1(100%) / 2(0%): 537 |
| DISCOVERY_DATE | 2345 | 0.0 | 2017-01-30 | 150 | 0.2939 | 2017-01-30: 150; 2015-02-14: 131; 2018-07-04: 127; 2016-02-08: 126; 2017-01-31: 115 |
| CONT_DATE | 1430 | 89.9105 | 2/8/2016 0:00:00 | 37 | 0.0725 | 2/8/2016 0:00:00: 37; 10/18/2015 0:00:00: 28; 1/30/2017 0:00:00: 26; 1/31/2017 0:00:00: 24; 10/17/2015 0:00:00: 24 |
| FIRE_CODE | 614 | 98.4912 | D1CZ | 104 | 0.2038 | D1CZ: 104; EK28: 31; J12X: 15; J0WR: 5; EKV2: 3 |
| ICS_209_PLUS_INCIDENT_JOIN_ID | 442 | 99.1241 | 2018_9058300_ARCHER CITY COMPLEX | 6 | 0.0118 | 2018_9058300_ARCHER CITY COMPLEX: 6; 2014_283614_MOODY: 1; 2018_9214716_TRICO: 1; 2018_9025438_FOLLEY PARK: 1; 2018_9024749_412: 1 |
| COUNTY | 288 | 0.5114 | Upshur | 1438 | 2.8178 | Upshur: 1438; McLennan: 1354; Tarrant: 1353; Hill: 1323; Dallas: 1166 |
| LatLong_County | 261 | 0.0294 | Upshur | 1440 | 2.8217 | Upshur: 1440; McLennan: 1355; Tarrant: 1353; Hill: 1322; Dallas: 1166 |
| FIPS_NAME | 251 | 0.5114 | Upshur County | 1438 | 2.8178 | Upshur County: 1438; McLennan County: 1354; Tarrant County: 1353; Hill County: 1323; Dallas County: 1166 |



## 11. Leakage Analysis

### Summary
| Leakage Label | Column Count |
| --- | --- |
| Definite Leakage | 9 |
| Definitely Safe | 294 |
| Possible Leakage | 6 |


### Definite Leakage Columns
- `ICS_209_PLUS_INCIDENT_JOIN_ID`
- `ICS_209_PLUS_COMPLEX_JOIN_ID`
- `MTBS_ID`
- `MTBS_FIRE_NAME`
- `CONT_DATE`
- `CONT_DOY`
- `CONT_TIME`
- `FIRE_SIZE`
- `FIRE_SIZE_CLASS`

> These columns contain information that would only be available **after** a fire event, and must be excluded from any predictive model.

### Possible Leakage Columns
- `LOCAL_FIRE_REPORT_ID`
- `FIRE_CODE`
- `FIRE_NAME`
- `COMPLEX_NAME`
- `DISCOVERY_DOY`
- `DISCOVERY_TIME`

> These columns require domain expert review before inclusion.



## 12. Predictive Readiness

### Summary by Label
| Readiness Label | Column Count |
| --- | --- |
| Administrative | 18 |
| Candidate Feature | 38 |
| Likely Remove | 21 |
| Review Later | 232 |


### Candidate Features (pre-fire, not leakage)
- `FIRE_YEAR`
- `LATITUDE`
- `LONGITUDE`
- `MOD_NDVI_12m`
- `MOD_EVI_12m`
- `Population`
- `Popo_1km`
- `bi_Normal`
- `erc_Normal`
- `No_FireStation_5.0km`
- `No_FireStation_10.0km`
- `No_FireStation_20.0km`
- `Aspect_1km`
- `Elevation_1km`
- `Elevation`
- `Slope_1km`
- `Aspect`
- `Slope`
- `EPL_MOBILE`
- `road_common_name_dis`
- `bi`
- `erc`
- `bi_5D_mean`
- `erc_5D_mean`
- `bi_5D_max`
- `erc_5D_max`
- `sph_Percentile`
- `vs_Percentile`
- `fm100_Percentile`
- `bi_Percentile`

> **Important:** This is a recommendation only. Final feature selection requires domain expert review.



## 13. Feature Dependency Groups

_No data available._



## 14. Source Readiness (Update Frequency)

| Update Frequency | Column Count |
| --- | --- |
| Administrative | 26 |
| Daily | 28 |
| Event Based | 18 |
| Monthly | 6 |
| Static | 30 |
| Unknown | 201 |



## 15. Recommended Next Steps

Based on the exploratory analysis, the following actions are recommended:

### 🔴 Immediate Actions (Pre-Modeling)
1. **Remove Definite Leakage Columns** — Exclude `CONT_DATE`, `CONT_DOY`, `CONT_TIME`, 
   `MTBS_ID`, `MTBS_FIRE_NAME`, `ICS_209_PLUS_*`, `FIRE_SIZE`, `FIRE_SIZE_CLASS` 
   from input features (they are post-fire outcomes, not predictors).
2. **Review High-Missing Columns** — Columns with >90% missing data (e.g., `geometry`, 
   `road_interstate_dis`, `GACC_Fire Use Teams`) should be evaluated for drop or imputation.
3. **Resolve Constant/Near-Constant Columns** — Zero-variance columns contribute no 
   information and can be safely dropped after confirmation.
4. **Investigate Duplicate Columns** — Review duplicate-content groups and keep only 
   the most interpretable representative.

### 🟡 Feature Engineering Decisions
5. **Address Skewed Features** — Highly skewed numeric columns (|skew| > 2) may benefit 
   from log or Box-Cox transformations in the preprocessing phase.
6. **Resolve High-Cardinality Categoricals** — Columns with >100 unique categories may 
   need target encoding, hashing, or grouping of rare levels.
7. **Verify Temporal Features** — Confirm `DISCOVERY_DOY`, `DISCOVERY_DATE` month/season 
   encoding strategy before training.

### 🟢 Modeling Readiness
8. **Define Target Variable** — Confirm whether target is binary ignition (Yes/No), 
   fire size class (A–G), or continuous acres burned.
9. **Finalize Feature Set** — After addressing leakage and quality issues, compile the 
   final feature matrix from Candidate Feature columns.
10. **Train/Test Split Strategy** — Use temporal splitting (train on 2014–2018, 
    validate on 2019, test on 2020) to prevent temporal leakage.

### 📊 Additional Analysis
11. **Spatial Autocorrelation** — Run Moran's I on fire occurrence to understand 
    spatial clustering patterns.
12. **Class Imbalance Assessment** — Analyze the ratio of fire vs. non-fire grid cells 
    if framing as a classification problem.


---

_Report generated by the Wildfire Dataset Analysis Pipeline — 2026-07-03 10:01:51_
