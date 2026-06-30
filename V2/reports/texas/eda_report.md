# Texas - Exploratory Data Analysis Report

_Phase-1 Pipeline | State: TX_


## 1. Dataset Overview

| Property | Value |
|----------|-------|
| State | Texas (TX) |
| Rows | 51,033 |
| Columns | 309 |
| Memory usage | 281.8 MB |

## 2. Column Data Types

| Dtype | Column Count |
|-------|-------------|
| `float64` | 251 |
| `object` | 54 |
| `int64` | 4 |

- Numeric columns    : **255**
- Categorical/object : **54**

## 3. Missing Values

- Columns with >=1 missing value : **195** / 309
- Columns fully populated        : **114** / 309

**Top 30 columns by % missing:**

| Column | Missing % |
| --- | --- |
| GACC_Fire Use Teams | 100.0 |
| IALMIL_87 | 100.0 |
| geometry | 100.0 |
| IAPLHS_88 | 100.0 |
| IAULHS_89 | 100.0 |
| IAHSEF | 100.0 |
| ICS_209_PLUS_COMPLEX_JOIN_ID | 99.97 |
| COMPLEX_NAME | 99.95 |
| road_interstate_dis | 99.92 |
| road_US_dis | 99.78 |
| MTBS_ID | 99.68 |
| MTBS_FIRE_NAME | 99.68 |
| road_other_dis | 99.49 |
| NWCG_CAUSE_AGE_CATEGORY | 99.41 |
| ICS_209_PLUS_INCIDENT_JOIN_ID | 99.12 |
| LOCAL_FIRE_REPORT_ID | 99.12 |
| road_state_dis | 98.5 |
| Evacuation | 98.5 |
| FIRE_CODE | 98.49 |
| road_county_dis | 98.12 |
| No_FireStation_1.0km | 90.53 |
| CONT_TIME | 90.27 |
| DISCOVERY_TIME | 90.06 |
| CONT_DATE | 89.91 |
| CONT_DOY | 89.91 |
| No_FireStation_5.0km | 37.29 |
| road_common_name_dis | 33.7 |
| ExoticAnnualGrass | 33.0 |
| PoaSecunda | 33.0 |
| Medusahead | 33.0 |

## 4. Fire Size Distribution (acres)

| Size Class (acres) | Count | Pct |
|--------------------|-------|-----|
| <0.25 | 11,890 | 23.3% |
| 0.25-1 | 19,039 | 37.3% |
| 1-10 | 13,661 | 26.8% |
| 10-100 | 5,002 | 9.8% |
| 100-300 | 802 | 1.6% |
| 300-1k | 401 | 0.8% |
| 1k-5k | 182 | 0.4% |
| >5k | 56 | 0.1% |

**Percentiles:**

| Percentile | Value (acres) |
|-----------|--------------|
| 0th | 0.0100 |
| 5th | 0.0200 |
| 10th | 0.1000 |
| 25th | 0.5000 |
| 50th | 1.0000 |
| 75th | 4.0000 |
| 90th | 20.0000 |
| 95th | 50.0000 |
| 99th | 410.0000 |
| 100th | 318156.0000 |

## 5. Distribution of Fire Causes

| Value | Count |
| --- | --- |
| Debris and open burning | 20481 |
| Missing data/not specified/undetermined | 16615 |
| Equipment and vehicle use | 5968 |
| Power generation/transmission/distribution | 3367 |
| Natural | 1892 |
| Smoking | 836 |
| Arson/incendiarism | 742 |
| Recreation and ceremony | 501 |
| Misuse of fire by a minor | 300 |
| Railroad operations and maintenance | 239 |
| Other causes | 75 |
| Fireworks | 13 |
| Firearms and explosives use | 4 |

## 6. Fires per Year

| Value | Fire Count |
| --- | --- |
| 2014 | 8538 |
| 2015 | 8304 |
| 2016 | 8586 |
| 2017 | 8922 |
| 2018 | 7533 |
| 2019 | 6455 |
| 2020 | 2695 |

## 7. Fires per Month

| Value | Fire Count |
| --- | --- |
| 1 (Jan) | 5582 |
| 2 (Feb) | 4817 |
| 3 (Mar) | 4727 |
| 4 (Apr) | 3024 |
| 5 (May) | 3428 |
| 6 (Jun) | 4104 |
| 7 (Jul) | 6526 |
| 8 (Aug) | 6015 |
| 9 (Sep) | 4102 |
| 10 (Oct) | 4032 |
| 11 (Nov) | 2476 |
| 12 (Dec) | 2200 |

## 8. Top 20 Counties by Fire Count

| Value | Count |
| --- | --- |
| Upshur | 1438 |
| McLennan | 1354 |
| Tarrant | 1353 |
| Hill | 1323 |
| Dallas | 1166 |
| Rusk | 1033 |
| Cherokee | 931 |
| Taylor | 814 |
| Navarro | 740 |
| Erath | 700 |
| Victoria | 697 |
| Coryell | 689 |
| Grimes | 661 |
| Hunt | 651 |
| Starr | 645 |
| Smith | 618 |
| Harrison | 614 |
| Van Zandt | 555 |
| Bell | 536 |
| Atascosa | 535 |

## 9. Top 15 Reporting Agencies

| Value | Count |
| --- | --- |
| ST/C&L | 50251 |
| FWS | 303 |
| NPS | 233 |
| FS | 231 |
| TRIBE | 9 |
| BLM | 6 |

## 10. Correlation Matrix (numeric columns)

_Full matrix saved to `correlation.csv`. Preview (first 15x15):_

|  | FOD_ID | LOCAL_FIRE_REPORT_ID | FIRE_YEAR | DISCOVERY_DOY | DISCOVERY_TIME | CONT_DOY | CONT_TIME | FIRE_SIZE | LATITUDE | LONGITUDE | FIPS_CODE | NPL | GAP_Sts | GAP_Prity | EVH |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FOD_ID | 1.0 | 0.131 | 0.806 | -0.025 | 0.01 | -0.041 | -0.053 | 0.009 | -0.048 | -0.077 | 0.005 | 0.005 | -0.01 | 0.006 | 0.024 |
| LOCAL_FIRE_REPORT_ID | 0.131 | 1.0 | 0.207 | -0.026 | -0.011 | -0.03 | 0.051 | 0.073 | 0.719 | -0.65 | -0.067 | 0.019 | 0.291 | -0.463 | 0.278 |
| FIRE_YEAR | 0.806 | 0.207 | 1.0 | -0.021 | 0.038 | 0.014 | -0.046 | 0.007 | -0.057 | -0.092 | 0.02 | 0.024 | -0.01 | 0.009 | 0.016 |
| DISCOVERY_DOY | -0.025 | -0.026 | -0.021 | 1.0 | 0.021 | 1.0 | -0.019 | -0.008 | 0.005 | 0.051 | 0.001 | 0.357 | 0.003 | -0.006 | 0.023 |
| DISCOVERY_TIME | 0.01 | -0.011 | 0.038 | 0.021 | 1.0 | 0.023 | 0.27 | 0.002 | 0.024 | -0.045 | 0.004 | 0.051 | 0.124 | -0.126 | -0.007 |
| CONT_DOY | -0.041 | -0.03 | 0.014 | 1.0 | 0.023 | 1.0 | -0.022 | -0.026 | -0.009 | 0.008 | 0.002 | 0.336 | 0.024 | -0.028 | 0.056 |
| CONT_TIME | -0.053 | 0.051 | -0.046 | -0.019 | 0.27 | -0.022 | 1.0 | 0.006 | -0.032 | 0.03 | -0.005 | 0.021 | 0.071 | -0.076 | 0.016 |
| FIRE_SIZE | 0.009 | 0.073 | 0.007 | -0.008 | 0.002 | -0.026 | 0.006 | 1.0 | 0.022 | -0.033 | -0.002 | -0.003 | -0.02 | 0.015 | 0.013 |
| LATITUDE | -0.048 | 0.719 | -0.057 | 0.005 | 0.024 | -0.009 | -0.032 | 0.022 | 1.0 | -0.164 | 0.048 | -0.028 | 0.106 | -0.088 | 0.049 |
| LONGITUDE | -0.077 | -0.65 | -0.092 | 0.051 | -0.045 | 0.008 | 0.03 | -0.033 | -0.164 | 1.0 | 0.029 | -0.019 | 0.003 | 0.014 | -0.064 |
| FIPS_CODE | 0.005 | -0.067 | 0.02 | 0.001 | 0.004 | 0.002 | -0.005 | -0.002 | 0.048 | 0.029 | 1.0 | -0.014 | 0.02 | -0.014 | -0.039 |
| NPL | 0.005 | 0.019 | 0.024 | 0.357 | 0.051 | 0.336 | 0.021 | -0.003 | -0.028 | -0.019 | -0.014 | 1.0 | -0.018 | 0.013 | 0.025 |
| GAP_Sts | -0.01 | 0.291 | -0.01 | 0.003 | 0.124 | 0.024 | 0.071 | -0.02 | 0.106 | 0.003 | 0.02 | -0.018 | 1.0 | -0.828 | -0.053 |
| GAP_Prity | 0.006 | -0.463 | 0.009 | -0.006 | -0.126 | -0.028 | -0.076 | 0.015 | -0.088 | 0.014 | -0.014 | 0.013 | -0.828 | 1.0 | 0.029 |
| EVH | 0.024 | 0.278 | 0.016 | 0.023 | -0.007 | 0.056 | 0.016 | 0.013 | 0.049 | -0.064 | -0.039 | 0.025 | -0.053 | 0.029 | 1.0 |

## 11. Constant Columns (single unique value)

Found **9** constant column(s):

- `STATE` = `TX`
- `GACC_Fire Use Teams` = `nan`
- `IALMIL_87` = `nan`
- `IAPLHS_88` = `nan`
- `IAULHS_89` = `nan`
- `IAHSEF` = `nan`
- `tmmn_Percentile` = `>90%`
- `tmmx_Percentile` = `>90%`
- `geometry` = `nan`

## 12. Duplicate Columns

Found **8** group(s) of duplicate columns:

- ['FIRE_YEAR', 'source_year']
- ['GACC_Fire Use Teams', 'IALMIL_87', 'IAPLHS_88', 'IAULHS_89', 'IAHSEF', 'geometry']
- ['M_WTR', 'WDLI']
- ['SM_C', 'SM_PFS']
- ['WD_ET', 'M_WTR_EOMI']
- ['IA_LMI_ET', 'IA_UN_ET', 'IA_POV_ET', 'IAULHSE', 'IAPLHSE', 'IALMILHSE', 'IALHE']
- ['LHE', 'M_WKFC_105']
- ['tmmn_Percentile', 'tmmx_Percentile']

## 13. Highly Correlated Columns (|r| >= 0.95)

Found **102** highly correlated pair(s):

| Column A | Column B | |r| |
|----------|----------|------|
| FIRE_YEAR | source_year | 1.0 |
| M_WTR | WDLI | 1.0 |
| SM_C | SM_PFS | 1.0 |
| WD_ET | M_WTR_EOMI | 1.0 |
| LHE | M_WKFC_105 | 1.0 |
| RPL_THEMES | RPL_THEME1 | 1.0 |
| RPL_THEMES | RPL_THEME4 | 1.0 |
| RPL_THEME1 | RPL_THEME4 | 1.0 |
| EPL_POV | EPL_NOVEH | 1.0 |
| CheatGrass | ExoticAnnualGrass | 1.0 |
| CheatGrass | Medusahead | 1.0 |
| CheatGrass | PoaSecunda | 1.0 |
| Medusahead | PoaSecunda | 1.0 |
| DISCOVERY_DOY | CONT_DOY | 0.9999 |
| CA | NCA | 0.9999 |
| EPL_POV | EPL_MOBILE | 0.9999 |
| EPL_PCI | RPL_THEME3 | 0.9999 |
| EPL_AGE17 | EPL_SNGPNT | 0.9999 |
| EPL_AGE17 | EPL_MINRTY | 0.9999 |
| EPL_AGE17 | EPL_LIMENG | 0.9999 |
| EPL_AGE17 | EPL_CROWD | 0.9999 |
| EPL_SNGPNT | EPL_MINRTY | 0.9999 |
| EPL_SNGPNT | EPL_LIMENG | 0.9999 |
| EPL_SNGPNT | EPL_MUNIT | 0.9999 |
| EPL_SNGPNT | EPL_CROWD | 0.9999 |
| EPL_MINRTY | EPL_LIMENG | 0.9999 |
| EPL_MINRTY | EPL_MUNIT | 0.9999 |
| EPL_MINRTY | EPL_CROWD | 0.9999 |
| EPL_LIMENG | EPL_MUNIT | 0.9999 |
| EPL_LIMENG | EPL_CROWD | 0.9999 |
| EPL_MUNIT | EPL_CROWD | 0.9999 |
| EPL_MOBILE | EPL_NOVEH | 0.9999 |
| ExoticAnnualGrass | Medusahead | 0.9999 |
| ExoticAnnualGrass | PoaSecunda | 0.9999 |
| EPL_AGE65 | EPL_MINRTY | 0.9998 |
| EPL_AGE65 | EPL_MUNIT | 0.9998 |
| EPL_AGE65 | EPL_CROWD | 0.9998 |
| EPL_AGE17 | EPL_MUNIT | 0.9998 |
| EPL_MINRTY | EPL_GROUPQ | 0.9998 |
| EPL_MUNIT | EPL_GROUPQ | 0.9998 |
| Elevation_1km | Elevation | 0.9997 |
| EPL_AGE65 | EPL_AGE17 | 0.9997 |
| EPL_AGE65 | EPL_SNGPNT | 0.9997 |
| EPL_AGE65 | EPL_LIMENG | 0.9997 |
| EPL_AGE65 | EPL_GROUPQ | 0.9997 |
| EPL_AGE17 | EPL_GROUPQ | 0.9997 |
| EPL_SNGPNT | EPL_GROUPQ | 0.9997 |
| EPL_LIMENG | EPL_GROUPQ | 0.9997 |
| EPL_CROWD | EPL_GROUPQ | 0.9997 |
| LILHSE | LISO_ET | 0.9985 |

## 14. Categorical Column Cardinality

| Column | Unique Values | High Cardinality |
|--------|--------------|-----------------|
| FPA_ID | 51,033 | [HIGH] |
| LOCAL_INCIDENT_ID | 50,073 | [HIGH] |
| EVT_1km | 47,130 | [HIGH] |
| Land_Cover_1km | 46,587 | [HIGH] |
| EVH_1km | 44,470 | [HIGH] |
| MOD_EVI_12m | 41,236 | [HIGH] |
| MOD_NDVI_12m | 41,236 | [HIGH] |
| NDVI_max | 41,218 | [HIGH] |
| NDVI_mean | 41,218 | [HIGH] |
| NDVI_min | 41,160 | [HIGH] |
| EVC_1km | 41,117 | [HIGH] |
| FIRE_NAME | 37,489 | [HIGH] |
| FRG_1km | 11,435 | [HIGH] |
| DISCOVERY_DATE | 2,345 | [HIGH] |
| CONT_DATE | 1,431 | [HIGH] |
| FIRE_CODE | 615 | [HIGH] |
| ICS_209_PLUS_INCIDENT_JOIN_ID | 443 | [HIGH] |
| COUNTY | 289 | [HIGH] |
| LatLong_County | 262 | [HIGH] |
| FIPS_NAME | 252 | [HIGH] |
| MTBS_ID | 160 | [HIGH] |
| MTBS_FIRE_NAME | 158 | [HIGH] |
| Ecoregion_US_L4CODE | 57 | [HIGH] |
| SOURCE_REPORTING_UNIT_NAME | 46 | No |
| SOURCE_REPORTING_UNIT | 39 | No |
| NWCG_REPORTING_UNIT_NAME | 33 | No |
| NWCG_REPORTING_UNIT_ID | 33 | No |
| Des_Tp | 31 | No |
| NAME | 22 | No |
| Mang_Name | 21 | No |
| COMPLEX_NAME | 19 | No |
| OWNER_DESCR | 13 | No |
| vpd_Percentile | 13 | No |
| NWCG_GENERAL_CAUSE | 13 | No |
| Ecoregion_NA_L3CODE | 13 | No |
| ICS_209_PLUS_COMPLEX_JOIN_ID | 11 | No |
| GACCAbbrev | 11 | No |
| Mang_Type | 9 | No |
| bi_Percentile | 7 | No |
| FIRE_SIZE_CLASS | 7 | No |
| vs_Percentile | 7 | No |
| erc_Percentile | 7 | No |
| fm100_Percentile | 7 | No |
| SOURCE_SYSTEM | 6 | No |
| LatLong_State | 6 | No |
| sph_Percentile | 6 | No |
| NWCG_REPORTING_AGENCY | 6 | No |
| SOURCE_SYSTEM_TYPE | 3 | No |
| NWCG_CAUSE_CLASSIFICATION | 3 | No |
| UI_EXP | 2 | No |
| NWCG_CAUSE_AGE_CATEGORY | 2 | No |
| tmmx_Percentile | 1 | No |
| tmmn_Percentile | 1 | No |
| STATE | 1 | No |

## 15. Potential Useless Columns

- `FOD_ID` - all values unique (likely ID/index)
- `FPA_ID` - all values unique (likely ID/index)
- `LOCAL_FIRE_REPORT_ID` - 99% missing
- `FIRE_CODE` - 98% missing
- `ICS_209_PLUS_INCIDENT_JOIN_ID` - 99% missing
- `ICS_209_PLUS_COMPLEX_JOIN_ID` - 100% missing
- `MTBS_ID` - 100% missing
- `MTBS_FIRE_NAME` - 100% missing
- `COMPLEX_NAME` - 100% missing
- `NWCG_CAUSE_AGE_CATEGORY` - 99% missing
- `STATE` - constant (1 unique value)
- `GACC_Fire Use Teams` - constant (1 unique value)
- `IALMIL_87` - constant (1 unique value)
- `IAPLHS_88` - constant (1 unique value)
- `IAULHS_89` - constant (1 unique value)
- `IAHSEF` - constant (1 unique value)
- `road_county_dis` - 98% missing
- `road_interstate_dis` - 100% missing
- `road_other_dis` - 99% missing
- `road_state_dis` - 99% missing
- `road_US_dis` - 100% missing
- `Evacuation` - 98% missing
- `tmmn_Percentile` - constant (1 unique value)
- `tmmx_Percentile` - constant (1 unique value)
- `geometry` - constant (1 unique value)

## 16. Potential Leakage Columns

_These columns describe post-ignition outcomes and should be excluded from any predictive model trained to predict ignition._

- `ICS_209_PLUS_INCIDENT_JOIN_ID`
- `ICS_209_PLUS_COMPLEX_JOIN_ID`
- `MTBS_ID`
- `MTBS_FIRE_NAME`
- `CONT_DATE`
- `CONT_DOY`
- `CONT_TIME`