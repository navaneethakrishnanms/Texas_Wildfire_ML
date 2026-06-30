# California - Exploratory Data Analysis Report

_Phase-1 Pipeline | State: CA_


## 1. Dataset Overview

| Property | Value |
|----------|-------|
| State | California (CA) |
| Rows | 50,881 |
| Columns | 309 |
| Memory usage | 283.7 MB |

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
| IALMIL_87 | 100.0 |
| IAULHS_89 | 100.0 |
| geometry | 100.0 |
| GACC_Fire Use Teams | 100.0 |
| IAHSEF | 100.0 |
| IAPLHS_88 | 100.0 |
| road_interstate_dis | 99.92 |
| road_US_dis | 99.76 |
| ICS_209_PLUS_COMPLEX_JOIN_ID | 99.57 |
| COMPLEX_NAME | 99.56 |
| MTBS_ID | 99.4 |
| MTBS_FIRE_NAME | 99.4 |
| road_other_dis | 99.32 |
| road_state_dis | 98.64 |
| ICS_209_PLUS_INCIDENT_JOIN_ID | 98.53 |
| road_county_dis | 98.35 |
| NWCG_CAUSE_AGE_CATEGORY | 97.68 |
| Evacuation | 89.13 |
| No_FireStation_1.0km | 84.04 |
| FIRE_CODE | 76.7 |
| LOCAL_FIRE_REPORT_ID | 75.62 |
| WF_PFS | 46.3 |
| CONT_TIME | 40.86 |
| CONT_DATE | 40.24 |
| CONT_DOY | 40.24 |
| road_common_name_dis | 35.02 |
| FIRE_NAME | 30.35 |
| CheatGrass | 27.21 |
| ExoticAnnualGrass | 27.21 |
| PoaSecunda | 27.21 |

## 4. Fire Size Distribution (acres)

| Size Class (acres) | Count | Pct |
|--------------------|-------|-----|
| <0.25 | 32,867 | 64.6% |
| 0.25-1 | 9,855 | 19.4% |
| 1-10 | 5,465 | 10.7% |
| 10-100 | 1,819 | 3.6% |
| 100-300 | 356 | 0.7% |
| 300-1k | 202 | 0.4% |
| 1k-5k | 181 | 0.4% |
| >5k | 136 | 0.3% |

**Percentiles:**

| Percentile | Value (acres) |
|-----------|--------------|
| 0th | 0.0100 |
| 5th | 0.0100 |
| 10th | 0.0100 |
| 25th | 0.1000 |
| 50th | 0.1000 |
| 75th | 1.0000 |
| 90th | 3.0000 |
| 95th | 12.0000 |
| 99th | 314.4000 |
| 100th | 410203.0000 |

## 5. Distribution of Fire Causes

| Value | Count |
| --- | --- |
| Missing data/not specified/undetermined | 25786 |
| Debris and open burning | 5233 |
| Equipment and vehicle use | 4948 |
| Natural | 4909 |
| Arson/incendiarism | 4234 |
| Recreation and ceremony | 1909 |
| Misuse of fire by a minor | 1173 |
| Smoking | 1149 |
| Power generation/transmission/distribution | 1044 |
| Other causes | 222 |
| Fireworks | 142 |
| Firearms and explosives use | 99 |
| Railroad operations and maintenance | 33 |

## 6. Fires per Year

| Value | Fire Count |
| --- | --- |
| 2014 | 6493 |
| 2015 | 7354 |
| 2016 | 7882 |
| 2017 | 9537 |
| 2018 | 9488 |
| 2019 | 6454 |
| 2020 | 3673 |

## 7. Fires per Month

| Value | Fire Count |
| --- | --- |
| 1 (Jan) | 1215 |
| 2 (Feb) | 1528 |
| 3 (Mar) | 1447 |
| 4 (Apr) | 2968 |
| 5 (May) | 5916 |
| 6 (Jun) | 9076 |
| 7 (Jul) | 9745 |
| 8 (Aug) | 6703 |
| 9 (Sep) | 5304 |
| 10 (Oct) | 3929 |
| 11 (Nov) | 2129 |
| 12 (Dec) | 921 |

## 8. Top 20 Counties by Fire Count

| Value | Count |
| --- | --- |
| RIVERSIDE | 4775 |
| FRESNO | 3103 |
| MERCED | 1966 |
| MADERA | 1371 |
| SAN BERNARDINO | 1349 |
| BUTTE | 1295 |
| SAN DIEGO | 1195 |
| PLACER | 1156 |
| LOS ANGELES | 1064 |
| EL DORADO | 966 |
| SAN LUIS OBISPO | 907 |
| KERN | 837 |
| SHASTA | 822 |
| CONTRA COSTA | 802 |
| SACRAMENTO | 764 |
| STANISLAUS | 763 |
| TEHAMA | 738 |
| MENDOCINO | 711 |
| SANTA CLARA | 708 |
| 093 | 702 |

## 9. Top 15 Reporting Agencies

| Value | Count |
| --- | --- |
| ST/C&L | 38960 |
| FS | 7979 |
| BLM | 1705 |
| BIA | 1203 |
| NPS | 835 |
| FWS | 168 |
| DOD | 26 |
| BOR | 5 |

## 10. Correlation Matrix (numeric columns)

_Full matrix saved to `correlation.csv`. Preview (first 15x15):_

|  | FOD_ID | LOCAL_FIRE_REPORT_ID | FIRE_YEAR | DISCOVERY_DOY | DISCOVERY_TIME | CONT_DOY | CONT_TIME | FIRE_SIZE | LATITUDE | LONGITUDE | FIPS_CODE | NPL | GAP_Sts | GAP_Prity | EVH |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FOD_ID | 1.0 | -0.413 | 0.788 | 0.061 | 0.004 | 0.105 | 0.01 | 0.001 | -0.138 | 0.094 | -0.014 | 0.069 | 0.114 | -0.134 | -0.122 |
| LOCAL_FIRE_REPORT_ID | -0.413 | 1.0 | -0.346 | -0.223 | -0.029 | -0.208 | -0.051 | -0.013 | -0.036 | -0.082 | -0.032 | -0.156 | 0.287 | -0.45 | -0.196 |
| FIRE_YEAR | 0.788 | -0.346 | 1.0 | -0.023 | 0.002 | 0.085 | 0.021 | -0.002 | -0.135 | 0.098 | -0.007 | -0.025 | 0.102 | -0.124 | -0.123 |
| DISCOVERY_DOY | 0.061 | -0.223 | -0.023 | 1.0 | -0.029 | 0.995 | -0.021 | 0.016 | 0.063 | -0.055 | -0.003 | 0.293 | -0.057 | 0.054 | 0.016 |
| DISCOVERY_TIME | 0.004 | -0.029 | 0.002 | -0.029 | 1.0 | -0.03 | 0.693 | 0.008 | -0.006 | 0.012 | -0.0 | 0.004 | -0.012 | 0.008 | 0.019 |
| CONT_DOY | 0.105 | -0.208 | 0.085 | 0.995 | -0.03 | 1.0 | -0.023 | 0.035 | 0.048 | -0.044 | 0.007 | 0.286 | -0.072 | 0.059 | 0.006 |
| CONT_TIME | 0.01 | -0.051 | 0.021 | -0.021 | 0.693 | -0.023 | 1.0 | 0.002 | 0.015 | 0.007 | 0.018 | 0.011 | -0.048 | 0.045 | 0.06 |
| FIRE_SIZE | 0.001 | -0.013 | -0.002 | 0.016 | 0.008 | 0.035 | 0.002 | 1.0 | 0.014 | -0.015 | 0.01 | 0.022 | -0.022 | 0.023 | 0.045 |
| LATITUDE | -0.138 | -0.036 | -0.135 | 0.063 | -0.006 | 0.048 | 0.015 | 0.014 | 1.0 | -0.87 | -0.009 | 0.14 | -0.11 | 0.168 | 0.156 |
| LONGITUDE | 0.094 | -0.082 | 0.098 | -0.055 | 0.012 | -0.044 | 0.007 | -0.015 | -0.87 | 1.0 | 0.004 | -0.099 | -0.039 | -0.008 | -0.038 |
| FIPS_CODE | -0.014 | -0.032 | -0.007 | -0.003 | -0.0 | 0.007 | 0.018 | 0.01 | -0.009 | 0.004 | 1.0 | -0.008 | -0.06 | 0.062 | 0.038 |
| NPL | 0.069 | -0.156 | -0.025 | 0.293 | 0.004 | 0.286 | 0.011 | 0.022 | 0.14 | -0.099 | -0.008 | 1.0 | -0.109 | 0.108 | 0.065 |
| GAP_Sts | 0.114 | 0.287 | 0.102 | -0.057 | -0.012 | -0.072 | -0.048 | -0.022 | -0.11 | -0.039 | -0.06 | -0.109 | 1.0 | -0.828 | -0.332 |
| GAP_Prity | -0.134 | -0.45 | -0.124 | 0.054 | 0.008 | 0.059 | 0.045 | 0.023 | 0.168 | -0.008 | 0.062 | 0.108 | -0.828 | 1.0 | 0.381 |
| EVH | -0.122 | -0.196 | -0.123 | 0.016 | 0.019 | 0.006 | 0.06 | 0.045 | 0.156 | -0.038 | 0.038 | 0.065 | -0.332 | 0.381 | 1.0 |

## 11. Constant Columns (single unique value)

Found **9** constant column(s):

- `STATE` = `CA`
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

Found **97** highly correlated pair(s):

| Column A | Column B | |r| |
|----------|----------|------|
| FIRE_YEAR | source_year | 1.0 |
| M_WTR | WDLI | 1.0 |
| SM_C | SM_PFS | 1.0 |
| WD_ET | M_WTR_EOMI | 1.0 |
| LHE | M_WKFC_105 | 1.0 |
| EPL_MINRTY | EPL_LIMENG | 1.0 |
| EPL_AGE17 | EPL_SNGPNT | 0.9999 |
| EPL_AGE17 | EPL_MINRTY | 0.9999 |
| EPL_AGE17 | EPL_LIMENG | 0.9999 |
| EPL_AGE17 | EPL_CROWD | 0.9999 |
| EPL_SNGPNT | EPL_MINRTY | 0.9999 |
| EPL_SNGPNT | EPL_LIMENG | 0.9999 |
| EPL_SNGPNT | EPL_CROWD | 0.9999 |
| EPL_MINRTY | EPL_MUNIT | 0.9999 |
| EPL_MINRTY | EPL_CROWD | 0.9999 |
| EPL_LIMENG | EPL_CROWD | 0.9999 |
| Ecoregion_NA_L2CODE | Ecoregion_NA_L1CODE | 0.9999 |
| CA | NCA | 0.9998 |
| EPL_AGE17 | EPL_MUNIT | 0.9998 |
| EPL_SNGPNT | EPL_MUNIT | 0.9998 |
| EPL_MINRTY | EPL_GROUPQ | 0.9998 |
| EPL_LIMENG | EPL_MUNIT | 0.9998 |
| EPL_MUNIT | EPL_CROWD | 0.9998 |
| EPL_MUNIT | EPL_GROUPQ | 0.9998 |
| Medusahead | PoaSecunda | 0.9998 |
| EPL_AGE65 | EPL_MUNIT | 0.9997 |
| EPL_AGE65 | EPL_GROUPQ | 0.9997 |
| EPL_AGE17 | EPL_GROUPQ | 0.9997 |
| EPL_SNGPNT | EPL_GROUPQ | 0.9997 |
| EPL_LIMENG | EPL_GROUPQ | 0.9997 |
| EPL_CROWD | EPL_GROUPQ | 0.9997 |
| CheatGrass | Medusahead | 0.9997 |
| CheatGrass | PoaSecunda | 0.9997 |
| EPL_AGE65 | EPL_SNGPNT | 0.9996 |
| EPL_AGE65 | EPL_MINRTY | 0.9996 |
| EPL_AGE65 | EPL_LIMENG | 0.9996 |
| EPL_AGE65 | EPL_CROWD | 0.9996 |
| CheatGrass | ExoticAnnualGrass | 0.9996 |
| EPL_AGE65 | EPL_AGE17 | 0.9995 |
| ExoticAnnualGrass | Medusahead | 0.9994 |
| ExoticAnnualGrass | PoaSecunda | 0.9991 |
| fm1000 | fm1000_5D_mean | 0.9989 |
| Elevation_1km | Elevation | 0.9984 |
| EVH | EVC | 0.9973 |
| fm1000_5D_mean | fm1000_5D_min | 0.9972 |
| TRI_1km | Slope_1km | 0.996 |
| fm1000 | fm1000_5D_min | 0.9958 |
| DISCOVERY_DOY | CONT_DOY | 0.9952 |
| FPL200S | M_EBSI | 0.9915 |
| ALI | A_ET | 0.9911 |

## 14. Categorical Column Cardinality

| Column | Unique Values | High Cardinality |
|--------|--------------|-----------------|
| FPA_ID | 50,881 | [HIGH] |
| EVT_1km | 41,996 | [HIGH] |
| EVH_1km | 40,186 | [HIGH] |
| LOCAL_INCIDENT_ID | 40,011 | [HIGH] |
| Land_Cover_1km | 39,929 | [HIGH] |
| EVC_1km | 39,786 | [HIGH] |
| MOD_NDVI_12m | 38,769 | [HIGH] |
| MOD_EVI_12m | 38,758 | [HIGH] |
| NDVI_max | 38,691 | [HIGH] |
| NDVI_mean | 38,691 | [HIGH] |
| NDVI_min | 38,690 | [HIGH] |
| FRG_1km | 24,591 | [HIGH] |
| FIRE_NAME | 17,713 | [HIGH] |
| FIRE_CODE | 7,704 | [HIGH] |
| DISCOVERY_DATE | 2,320 | [HIGH] |
| CONT_DATE | 2,143 | [HIGH] |
| ICS_209_PLUS_INCIDENT_JOIN_ID | 603 | [HIGH] |
| MTBS_ID | 290 | [HIGH] |
| MTBS_FIRE_NAME | 267 | [HIGH] |
| COUNTY | 178 | [HIGH] |
| Ecoregion_US_L4CODE | 174 | [HIGH] |
| SOURCE_REPORTING_UNIT_NAME | 160 | [HIGH] |
| SOURCE_REPORTING_UNIT | 142 | [HIGH] |
| NWCG_REPORTING_UNIT_ID | 122 | [HIGH] |
| NWCG_REPORTING_UNIT_NAME | 122 | [HIGH] |
| LatLong_County | 67 | [HIGH] |
| FIPS_NAME | 59 | [HIGH] |
| Des_Tp | 49 | No |
| COMPLEX_NAME | 35 | No |
| ICS_209_PLUS_COMPLEX_JOIN_ID | 33 | No |
| Mang_Name | 26 | No |
| NAME | 19 | No |
| OWNER_DESCR | 16 | No |
| NWCG_GENERAL_CAUSE | 13 | No |
| Ecoregion_NA_L3CODE | 13 | No |
| vpd_Percentile | 13 | No |
| GACCAbbrev | 11 | No |
| Mang_Type | 10 | No |
| NWCG_REPORTING_AGENCY | 8 | No |
| vs_Percentile | 7 | No |
| SOURCE_SYSTEM | 7 | No |
| FIRE_SIZE_CLASS | 7 | No |
| erc_Percentile | 7 | No |
| bi_Percentile | 7 | No |
| fm100_Percentile | 7 | No |
| sph_Percentile | 6 | No |
| LatLong_State | 5 | No |
| SOURCE_SYSTEM_TYPE | 3 | No |
| NWCG_CAUSE_CLASSIFICATION | 3 | No |
| NWCG_CAUSE_AGE_CATEGORY | 2 | No |
| UI_EXP | 2 | No |
| STATE | 1 | No |
| tmmx_Percentile | 1 | No |
| tmmn_Percentile | 1 | No |

## 15. Potential Useless Columns

- `FOD_ID` - all values unique (likely ID/index)
- `FPA_ID` - all values unique (likely ID/index)
- `ICS_209_PLUS_INCIDENT_JOIN_ID` - 99% missing
- `ICS_209_PLUS_COMPLEX_JOIN_ID` - 100% missing
- `MTBS_ID` - 99% missing
- `MTBS_FIRE_NAME` - 99% missing
- `COMPLEX_NAME` - 100% missing
- `NWCG_CAUSE_AGE_CATEGORY` - 98% missing
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