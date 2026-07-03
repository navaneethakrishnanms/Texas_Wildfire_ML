# Wildfire Dataset — Complete Exploratory Analysis Report

> **Generated:** 2026-07-03 10:10:19  
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
| **State** | California |
| **Total Rows** | 50,881 |
| **Total Columns** | 309 |
| **Memory Usage (MB)** | 280.76 MB |
| **Dtype Breakdown** | float64: 251; object: 54; int64: 4 |
| **Numeric Columns** | 255 |
| **Categorical Columns** | 35 |
| **Datetime Columns** | 0 |
| **Boolean Columns** | 0 |
| **Object Columns (raw)** | 54 |
| **Text Columns** | 19 |
| **Float64 Columns** | 251 |
| **Int64 Columns** | 4 |
| **Year Range** | 2014 – 2020 |
| **Data Source** | FPA-FOD Pre-processed (Phase-1 Pipeline) |



## 2. Schema & Feature Summary

- Total features analyzed: **309**

### Semantic Type Breakdown
- **Numeric**: 255 columns
- **Categorical**: 35 columns
- **Text**: 19 columns

### Sample — First 20 Features
| Column Name | Data Type | Semantic Type | Unique Values | Missing % | Description |
| --- | --- | --- | --- | --- | --- |
| FOD_ID | int64 | Numeric | 50881 | 0.0 | Fire occurrence database unique identifier |
| FPA_ID | object | Text | 50881 | 0.0 | No description available |
| SOURCE_SYSTEM_TYPE | object | Categorical | 3 | 0.0 | No description available |
| SOURCE_SYSTEM | object | Categorical | 7 | 0.0 | No description available |
| NWCG_REPORTING_AGENCY | object | Categorical | 8 | 0.0 | Agency that reported the fire |
| NWCG_REPORTING_UNIT_ID | object | Categorical | 122 | 0.0 | No description available |
| NWCG_REPORTING_UNIT_NAME | object | Categorical | 122 | 0.0 | No description available |
| SOURCE_REPORTING_UNIT | object | Categorical | 142 | 0.0 | No description available |
| SOURCE_REPORTING_UNIT_NAME | object | Categorical | 160 | 0.0 | No description available |
| LOCAL_FIRE_REPORT_ID | float64 | Numeric | 3417 | 75.6176 | No description available |
| LOCAL_INCIDENT_ID | object | Text | 40010 | 8.9542 | No description available |
| FIRE_CODE | object | Text | 7703 | 76.6966 | No description available |
| FIRE_NAME | object | Text | 17712 | 30.3532 | Name assigned to the fire |
| ICS_209_PLUS_INCIDENT_JOIN_ID | object | Text | 602 | 98.5279 | Join key to ICS-209+ incident database |
| ICS_209_PLUS_COMPLEX_JOIN_ID | object | Categorical | 32 | 99.5696 | Join key to ICS-209+ complex database |
| MTBS_ID | object | Text | 289 | 99.4045 | Monitoring Trends in Burn Severity fire ID |
| MTBS_FIRE_NAME | object | Text | 266 | 99.4045 | Fire name in MTBS dataset |
| COMPLEX_NAME | object | Categorical | 34 | 99.5598 | Name of fire complex (if part of multi-fire event) |
| FIRE_YEAR | int64 | Numeric | 7 | 0.0 | Calendar year the fire was discovered |
| DISCOVERY_DATE | object | Text | 2320 | 0.0 | Date fire was first reported |

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
| Critical (≥90.0%) | 17 |
| High (≥75.0%) | 4 |
| Low (≥25.0%) | 10 |
| Minimal (<25%) | 164 |


### Critical (≥90.0% missing)
  - `IALMIL_87`
  - `IAULHS_89`
  - `geometry`
  - `GACC_Fire Use Teams`
  - `IAHSEF`
  - `IAPLHS_88`
  - `road_interstate_dis`
  - `road_US_dis`
  - `ICS_209_PLUS_COMPLEX_JOIN_ID`
  - `COMPLEX_NAME`
  - `MTBS_ID`
  - `MTBS_FIRE_NAME`
  - `road_other_dis`
  - `road_state_dis`
  - `ICS_209_PLUS_INCIDENT_JOIN_ID`
  - `road_county_dis`
  - `NWCG_CAUSE_AGE_CATEGORY`

### High (≥75.0% missing)
  - `Evacuation`
  - `No_FireStation_1.0km`
  - `FIRE_CODE`
  - `LOCAL_FIRE_REPORT_ID`

### Moderate (≥50.0% missing)
  _None_

> **Note:** No values were imputed. No columns were removed. This is a read-only analysis.



## 5. Feature Quality Summary

| Metric | Value |
|--------|-------|
| Total Columns | 309 |
| PASS (no issues) | N/A |
| REVIEW (has issues) | 69 |

- **Constant columns**: `NWCG_CAUSE_AGE_CATEGORY`, `STATE`, `GACC_Fire Use Teams`, `IA_LMI_ET`, `IA_UN_ET`, `IA_POV_ET`, `IAULHSE`, `IAPLHSE`, `IALMILHSE`, `IALMIL_87`, `IAPLHS_88`, `IAULHS_89`, `IALHE`, `IAHSEF`, `UI_EXP`, `THRHLD`, `tmmn_Percentile`, `tmmx_Percentile`, `geometry`
- **Duplicate columns**: `FIRE_YEAR`, `GACC_Fire Use Teams`, `M_WTR`, `SM_C`, `SM_PFS`, `WDLI`, `WD_ET`, `IA_LMI_ET`, `IA_UN_ET`, `IA_POV_ET`, `IAULHSE`, `IAPLHSE`, `IALMILHSE`, `IALMIL_87`, `IAPLHS_88`, `IAULHS_89`, `LHE`, `IALHE`, `IAHSEF`, `M_WTR_EOMI`
- **Columns with infinite values**: _None_



## 6. Statistical Summary

- Numeric columns analyzed: **255**
- Highly skewed columns (|skew| > 2): **102**

### Top 15 by Absolute Skewness
| Column | Skewness | Mean | Median | Std | Outlier % |
| --- | --- | --- | --- | --- | --- |
| SDI | -159.496081 | -1.337561552007739e+32 | 0.0 | 2.1334001925572395e+34 | 9.3335 |
| pr_5D_min | 106.255962 | 0.005354 | 0.0 | 0.20923 | 0.3164 |
| FRG | -93.625661 | 15.700733 | 3.0 | 59.358685 | 10.3477 |
| FIRE_SIZE | 74.969478 | 102.8205 | 0.1 | 3181.436145 | 9.023 |
| EPL_MINRTY | -53.105818 | 0.225961 | 0.5762 | 18.807123 | 0.0354 |
| EPL_MUNIT | -53.101301 | -0.04809 | 0.2956 | 18.802498 | 0.0354 |
| EPL_CROWD | -53.100306 | 0.30804 | 0.73 | 18.809319 | 0.0354 |
| EPL_LIMENG | -53.099159 | 0.253811 | 0.6561 | 18.808433 | 0.0354 |
| EPL_SNGPNT | -53.098398 | 0.096086 | 0.4085 | 18.805554 | 0.0354 |
| EPL_GROUPQ | -53.094998 | 0.196931 | 0.6214 | 18.807854 | 0.0354 |
| EPL_AGE65 | -53.094896 | 0.220288 | 0.6334 | 18.808305 | 0.0354 |
| EPL_AGE17 | -53.094886 | 0.131438 | 0.4336 | 18.806634 | 0.0354 |
| EPL_NOHSDP | -51.682246 | 0.200323 | 0.5738 | 19.322879 | 0.0374 |
| RPL_THEME3 | -46.973113 | 0.154347 | 0.6337 | 21.25889 | 0.0452 |
| EPL_PCI | -45.24158 | 0.028476 | 0.5125 | 22.068717 | 0.0488 |



## 7. Temporal Analysis

### Fires by Year
| Year | Fire Count | % of Total | Total Acres | Mean Acres | Median Acres |
| --- | --- | --- | --- | --- | --- |
| 2014.0 | 6493.0 | 12.76 | 547828.57 | 84.37218081010317 | 0.1 |
| 2015.0 | 7354.0 | 14.45 | 779074.74 | 105.93890943704106 | 0.1 |
| 2016.0 | 7882.0 | 15.49 | 574862.96 | 72.93364120781527 | 0.1 |
| 2017.0 | 9537.0 | 18.74 | 1360160.0 | 142.61927230785363 | 0.1 |
| 2018.0 | 9488.0 | 18.65 | 1635757.6 | 172.40278246205733 | 0.1 |
| 2019.0 | 6454.0 | 12.68 | 294620.97 | 45.64936008676789 | 0.1 |
| 2020.0 | 3673.0 | 7.22 | 39305.01 | 10.701064524911516 | 0.2 |


### Fires by Month
| Month | Fire Count | Month Name | % of Total |
| --- | --- | --- | --- |
| 1 | 1215 | Jan | 2.39 |
| 2 | 1528 | Feb | 3.0 |
| 3 | 1447 | Mar | 2.84 |
| 4 | 2968 | Apr | 5.83 |
| 5 | 5916 | May | 11.63 |
| 6 | 9076 | Jun | 17.84 |
| 7 | 9745 | Jul | 19.15 |
| 8 | 6703 | Aug | 13.17 |
| 9 | 5304 | Sep | 10.42 |
| 10 | 3929 | Oct | 7.72 |
| 11 | 2129 | Nov | 4.18 |
| 12 | 921 | Dec | 1.81 |


### Duration Statistics
| Mean Duration (days) | Median Duration (days) | Max Duration (days) | P90 Duration (days) | P95 Duration (days) | Fires with Duration | Fires Missing Duration |
| --- | --- | --- | --- | --- | --- | --- |
| 0.89 | 0.0 | 365.0 | 1.0 | 2.0 | 30409.0 | 20472.0 |



## 8. Geographic Analysis

### Top States by Fire Count
| State | Fire Count | % of Total |
| --- | --- | --- |
| CA | 50881 | 100.0 |


### Top 15 Counties
| County | Fire Count | % of Total |
| --- | --- | --- |
| RIVERSIDE | 4775 | 9.3846 |
| FRESNO | 3103 | 6.0985 |
| MERCED | 1966 | 3.8639 |
| MADERA | 1371 | 2.6945 |
| SAN BERNARDINO | 1349 | 2.6513 |
| BUTTE | 1295 | 2.5452 |
| SAN DIEGO | 1195 | 2.3486 |
| PLACER | 1156 | 2.272 |
| LOS ANGELES | 1064 | 2.0912 |
| EL DORADO | 966 | 1.8985 |
| SAN LUIS OBISPO | 907 | 1.7826 |
| KERN | 837 | 1.645 |
| SHASTA | 822 | 1.6155 |
| CONTRA COSTA | 802 | 1.5762 |
| SACRAMENTO | 764 | 1.5015 |


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
| EPL_MINRTY | EPL_LIMENG | 0.999951 |
| Ecoregion_NA_L2CODE | Ecoregion_NA_L1CODE | 0.999938 |
| EPL_AGE17 | EPL_MINRTY | 0.99993 |
| EPL_MINRTY | EPL_CROWD | 0.999927 |
| EPL_SNGPNT | EPL_MINRTY | 0.999922 |
| EPL_AGE17 | EPL_SNGPNT | 0.999911 |
| EPL_LIMENG | EPL_CROWD | 0.999895 |
| EPL_SNGPNT | EPL_CROWD | 0.999891 |
| EPL_AGE17 | EPL_LIMENG | 0.999885 |
| EPL_AGE17 | EPL_CROWD | 0.999882 |
| EPL_SNGPNT | EPL_LIMENG | 0.999873 |
| EPL_MINRTY | EPL_MUNIT | 0.99987 |
| EPL_SNGPNT | EPL_MUNIT | 0.999847 |
| Medusahead | PoaSecunda | 0.999831 |
| EPL_LIMENG | EPL_MUNIT | 0.99983 |


### Top 20 Highly Correlated Pairs (Spearman)
| Feature A | Feature B | Correlation |
| --- | --- | --- |
| WD_ET | M_WTR_EOMI | 1.0 |
| LHE | M_WKFC_105 | 1.0 |
| SM_C | SM_PFS | 1.0 |
| M_WTR | WDLI | 1.0 |
| FIRE_YEAR | source_year | 1.0 |
| CA | NCA | -0.999974 |
| Ecoregion_NA_L2CODE | Ecoregion_NA_L1CODE | 0.999912 |
| pr_5D_mean | pr_5D_max | 0.999535 |
| fm1000 | fm1000_5D_mean | 0.999152 |
| TRI_1km | Slope_1km | 0.998331 |
| EVH | EVC | 0.998229 |
| CheatGrass | ExoticAnnualGrass | 0.997921 |
| ExoticAnnualGrass | Medusahead | 0.997193 |
| fm1000_5D_mean | fm1000_5D_min | 0.996568 |
| Elevation_1km | Elevation | 0.99635 |
| DISCOVERY_DOY | CONT_DOY | 0.995489 |
| fm1000 | fm1000_5D_min | 0.995296 |
| Medusahead | PoaSecunda | 0.994897 |
| CheatGrass | Medusahead | 0.994569 |
| ExoticAnnualGrass | PoaSecunda | 0.994376 |



## 10. Categorical Analysis

| Column | Unique Count | Missing % | Top Category | Top Category Count | Top Category % | Top 5 Categories |
| --- | --- | --- | --- | --- | --- | --- |
| FPA_ID | 50881 | 0.0 | 2014CAIRS24109077 | 1 | 0.002 | 2014CAIRS24109077: 1; SFO-2018CACDFBDU811531: 1; SFO-2018CACDFTCU006483: 1; SFO-2018CACDFBEU002719: 1; SFO-2018CACDFRRU072060: 1 |
| EVT_1km | 41996 | 0.0 | 7299(29%) / 7914(23%) / 7297(22%) | 31 | 0.0609 | 7299(29%) / 7914(23%) / 7297(22%): 31; 9301(36%) / 7043(10%) / 7113(9%): 31; 7043(23%) / 7299(13%) / 7967(11%): 23; 7299(31%) / 7296(14%) / 7030(13%): 23; 7299(21%) / 9301(17%) / 7296(15%): 22 |
| EVH_1km | 40186 | 0.0 | 303(56%) / 118(20%) / 308(8%) | 31 | 0.0609 | 303(56%) / 118(20%) / 308(8%): 31; 25(29%) / 17(23%) / 23(22%): 31; 25(13%) / 303(10%) / 17(8%): 26; 25(31%) / 22(14%) / 14(6%): 23; 25(28%) / 22(16%) / 23(13%): 22 |
| LOCAL_INCIDENT_ID | 40010 | 8.9542 | 11 | 106 | 0.2083 | 11: 106; 10: 104; 002: 102; 003: 101; 004: 100 |
| Land_Cover_1km | 39929 | 0.0 | 52(100%) | 56 | 0.1101 | 52(100%): 56; 52(44%) / 42(27%) / 43(15%): 31; 42(99%) / 52(1%): 27; 42(97%) / 52(3%): 25; 82(95%) / 21(3%) / 22(2%): 24 |
| EVC_1km | 39786 | 0.0 | 355(51%) / 135(8%) / 325(8%) | 31 | 0.0609 | 355(51%) / 135(8%) / 325(8%): 31; 25(13%) / 17(8%) / 22(8%): 31; 25(29%) / 17(23%) / 23(22%): 31; 25(12%) / 22(8%) / 17(6%): 29; 25(31%) / 22(14%) / 14(6%): 23 |
| MOD_NDVI_12m | 38769 | 0.0 | '0.77' '0.76' '0.73' '0.73' '0.73' '0.62' '0.71' '0.7' '0.74' '0.74' '0.73' '0.74' | 25 | 0.0491 | '0.77' '0.76' '0.73' '0.73' '0.73' '0.62' '0.71' '0.7' '0.74' '0.74' '0.73' '0.74': 25; '0.77' '0.76' '0.73' '0.69' '0.74' '0.74' '0.75' '0.75' '0.74' '0.74' '0.76' '0.76': 23; '0.75' '0.77' '0.76' '0.73' '0.69' '0.74' '0.74' '0.75' '0.75' '0.74' '0.74' '0.76': 20; '0.75' '0.77' '0.76' '0.73' '0.73' '0.73' '0.62' '0.71' '0.7' '0.74' '0.74' '0.73': 19; '0.85' '0.85' '0.82' '0.82' '0.82' '0.81' '0.84' '0.86' '0.87' '0.73' '0.86' '0.86': 18 |
| MOD_EVI_12m | 38758 | 0.0 | '0.49' '0.41' '0.39' '0.3' '0.3' '0.28' '0.29' '0.37' '0.39' '0.44' '0.45' '0.44' | 25 | 0.0491 | '0.49' '0.41' '0.39' '0.3' '0.3' '0.28' '0.29' '0.37' '0.39' '0.44' '0.45' '0.44': 25; '0.45' '0.39' '0.32' '0.34' '0.3' '0.32' '0.36' '0.35' '0.45' '0.41' '0.43' '0.44': 23; '0.46' '0.45' '0.39' '0.32' '0.34' '0.3' '0.32' '0.36' '0.35' '0.45' '0.41' '0.43': 20; '0.52' '0.49' '0.41' '0.39' '0.3' '0.3' '0.28' '0.29' '0.37' '0.39' '0.44' '0.45': 19; '0.43' '0.48' '0.54' '0.54' '0.51' '0.45' '0.4' '0.39' '0.36' '0.25' '0.36' '0.41': 18 |
| NDVI_mean | 38691 | 0.0 | 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' | 95 | 0.1867 | 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan': 95; '0.53' '0.55' '0.32' '0.33' '0.2' '0.12' '0.54' '0.54' '0.64' '0.46' '0.62' '0.69': 25; '0.55' '0.45' '0.42' '0.29' '0.51' '0.5' '0.41' '0.62' '0.5' '0.59' '0.67' '0.63': 23; '0.72' '0.55' '0.45' '0.42' '0.29' '0.51' '0.5' '0.41' '0.62' '0.5' '0.59' '0.67': 20; '0.67' '0.53' '0.55' '0.32' '0.33' '0.2' '0.12' '0.54' '0.54' '0.64' '0.46' '0.62': 19 |
| NDVI_max | 38691 | 0.0 | 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' | 95 | 0.1867 | 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan': 95; '0.83' '0.87' '0.79' '0.78' '0.82' '0.8' '0.93' '0.76' '0.8' '0.87' '0.8' '0.8': 25; '0.83' '0.81' '0.87' '0.81' '0.93' '0.79' '0.84' '0.85' '0.85' '0.84' '0.84' '0.84': 23; '0.82' '0.83' '0.81' '0.87' '0.81' '0.93' '0.79' '0.84' '0.85' '0.85' '0.84' '0.84': 20; '0.8' '0.83' '0.87' '0.79' '0.78' '0.82' '0.8' '0.93' '0.76' '0.8' '0.87' '0.8': 19 |
| NDVI_min | 38690 | 0.0 | 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' | 95 | 0.1867 | 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan': 95; '-0.0' '0.04' '-0.01' '-0.01' '-0.03' '-0.01' '0.01' '0.04' '0.02' '0.01' '-0.01' '-0.01': 25; '0.02' '0.01' '-0.0' '-0.03' '-0.01' '-0.04' '-0.04' '-0.0' '-0.01' '0.04' '-0.04' '0.04': 23; '0.05' '0.02' '0.01' '-0.0' '-0.03' '-0.01' '-0.04' '-0.04' '-0.0' '-0.01' '0.04' '-0.04': 20; '0.14' '-0.0' '0.04' '-0.01' '-0.01' '-0.03' '-0.01' '0.01' '0.04' '0.02' '0.01' '-0.01': 19 |
| FRG_1km | 24591 | 0.0 | 4(100%) / 3(0%) | 305 | 0.5994 | 4(100%) / 3(0%): 305; 4(100%): 281; 5(100%): 227; 4(99%) / 3(1%): 212; 4(99%) / 3(1%) / 1(0%): 140 |
| FIRE_NAME | 17712 | 30.3532 | CREEK | 150 | 0.2948 | CREEK: 150; HIGHWAY: 122; RIVER: 114; OAK: 107; LAKE: 101 |
| FIRE_CODE | 7703 | 76.6966 | EK1W | 432 | 0.849 | EK1W: 432; EK1S: 397; EK15: 336; EK10: 318; EK13: 310 |
| DISCOVERY_DATE | 2320 | 0.0 | 2015-07-30 | 160 | 0.3145 | 2015-07-30: 160; 2018-07-04: 156; 2017-07-04: 150; 2016-07-04: 139; 2018-07-15: 137 |
| CONT_DATE | 2142 | 40.2351 | 7/4/2015 0:00:00 | 94 | 0.1847 | 7/4/2015 0:00:00: 94; 7/20/2014 0:00:00: 84; 7/8/2015 0:00:00: 84; 7/25/2017 0:00:00: 84; 7/21/2014 0:00:00: 80 |
| ICS_209_PLUS_INCIDENT_JOIN_ID | 602 | 98.5279 | 2017_7238965_MODOC JULY COMPLEX | 53 | 0.1042 | 2017_7238965_MODOC JULY COMPLEX: 53; 2017_7382007_ORLEANS COMPLEX: 22; 2015_2744562_RIVER COMPLEX: 13; 2017_7396234_ECLIPSE COMPLEX: 10; 2015_2753601_FORK COMPLEX: 10 |
| MTBS_ID | 289 | 99.4045 | CA4064212358620150731 | 4 | 0.0079 | CA4064212358620150731: 4; CA4179612337420140814: 4; CA4178612347520170812: 3; CA4091312343720150731: 2; CA3312011716020140514: 2 |
| MTBS_FIRE_NAME | 266 | 99.4045 | ROUTE COMPLEX | 4 | 0.0079 | ROUTE COMPLEX: 4; HAPPY CAMP COMPLEX: 4; CREEK: 3; SAND: 3; RANCH: 3 |
| COUNTY | 177 | 9.0387 | RIVERSIDE | 4775 | 9.3846 | RIVERSIDE: 4775; FRESNO: 3103; MERCED: 1966; MADERA: 1371; SAN BERNARDINO: 1349 |



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

| Feature Group | Column Count |
| --- | --- |
| Other | 195 |
| Fire Weather Index | 27 |
| Administrative | 26 |
| Fire Identity | 11 |
| Terrain | 10 |
| Vegetation Index | 6 |
| Ecoregion | 5 |
| Fire Outcome | 5 |
| Fire Stations | 4 |
| Discovery | 3 |
| Cause | 3 |
| Geographic Coordinates | 3 |
| Infrastructure | 3 |
| Vegetation Types | 3 |
| Land Cover | 2 |
| Population | 2 |
| Climate / EJ | 1 |



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

_Report generated by the Wildfire Dataset Analysis Pipeline — 2026-07-03 10:10:19_
