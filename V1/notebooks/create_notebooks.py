"""Creates the three analysis notebooks for the Texas Wildfire POC."""
import json
from pathlib import Path

def make_nb(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "cells": cells
    }

def md(src): return {"cell_type": "markdown", "id": "a1", "metadata": {}, "source": src}
def code(src): return {"cell_type": "code", "id": "b1", "metadata": {}, "source": src,
                       "execution_count": None, "outputs": []}

nb01 = make_nb([
    md("# 01 - Exploratory Data Analysis\nExplore raw GEE rasters and FIRMS fire data."),
    code("import rasterio\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n%matplotlib inline\nRAW = 'data/raw'"),
    md("## 1. NDVI Raster"),
    code("with rasterio.open(f'{RAW}/ndvi/Texas_NDVI_2024.tif') as src:\n    ndvi = src.read(1).astype('float32') * 0.0001\nprint('Shape:', ndvi.shape)\nprint('Min:', round(float(ndvi[ndvi != 0].min()),4), 'Max:', round(float(ndvi.max()),4))\nplt.figure(figsize=(12,6))\nplt.imshow(ndvi, cmap='RdYlGn', vmin=-0.1, vmax=0.9)\nplt.colorbar(label='NDVI')\nplt.title('Texas NDVI 2024')\nplt.show()"),
    md("## 2. FIRMS Fire Points"),
    code("df = pd.read_csv(f'{RAW}/firms/Texas_FIRMS_2024.csv', parse_dates=['acq_date'])\nprint(df.shape)\nprint(df.describe())\ndf['acq_date'].dt.month.value_counts().sort_index().plot(kind='bar', title='Fires by Month')\nplt.tight_layout()\nplt.show()"),
    md("## 3. Dataset Class Balance"),
    code("dataset = pd.read_csv('data/processed/wildfire_dataset.csv')\nprint(dataset.shape)\nprint(dataset['fire_label'].value_counts())\ndataset.hist(figsize=(18,12), bins=40)\nplt.tight_layout()\nplt.show()"),
])

nb02 = make_nb([
    md("# 02 - FIRMS Fire Detection Analysis\nAnalyze spatial and temporal patterns of NASA VIIRS fire detections."),
    code("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\n%matplotlib inline\ndf = pd.read_csv('data/raw/firms/Texas_FIRMS_2024.csv', parse_dates=['acq_date'])\nprint(df.head())\nprint(df.dtypes)"),
    md("## Fire Radiative Power Distribution"),
    code("plt.figure(figsize=(10,4))\nplt.hist(df['frp'], bins=50, color='#ff4e00', edgecolor='black', alpha=0.8)\nplt.xlabel('Fire Radiative Power (MW)')\nplt.ylabel('Count')\nplt.title('FRP Distribution - Texas 2024')\nplt.tight_layout()\nplt.show()"),
    md("## Spatial Distribution"),
    code("plt.figure(figsize=(12,8))\nsc = plt.scatter(df['longitude'], df['latitude'], c=df['frp'], cmap='hot', s=10, alpha=0.7)\nplt.colorbar(sc, label='FRP (MW)')\nplt.title('FIRMS Fire Detections - Texas 2024')\nplt.xlabel('Longitude')\nplt.ylabel('Latitude')\nplt.tight_layout()\nplt.show()"),
])

nb03 = make_nb([
    md("# 03 - Model Performance Analysis\nEvaluate the trained XGBoost wildfire risk classifier."),
    code("import pandas as pd\nimport numpy as np\nimport pickle\nimport json\nimport matplotlib.pyplot as plt\nfrom sklearn.metrics import roc_auc_score, average_precision_score, classification_report, ConfusionMatrixDisplay\n%matplotlib inline\n\nwith open('models/xgb_model.pkl', 'rb') as f: model = pickle.load(f)\nwith open('models/scaler.pkl', 'rb')  as f: scaler = pickle.load(f)\nwith open('models/imputer.pkl', 'rb') as f: imputer = pickle.load(f)\nwith open('models/features.json') as f: feats = json.load(f)\n\ntest = pd.read_csv('data/processed/test.csv')\nX = imputer.transform(test[feats].values)\nX = scaler.transform(X)\nproba = model.predict_proba(X)[:,1]\ny = test['fire_label'].values\nprint('ROC-AUC:', round(roc_auc_score(y, proba), 4))\nprint('PR-AUC:', round(average_precision_score(y, proba), 4))"),
    md("## Confusion Matrix"),
    code("threshold = 0.5\npred = (proba >= threshold).astype(int)\nprint(classification_report(y, pred, target_names=['No Fire','Fire']))\nConfusionMatrixDisplay.from_predictions(y, pred, display_labels=['No Fire','Fire'])\nplt.title('Confusion Matrix')\nplt.tight_layout()\nplt.show()"),
    md("## SHAP Feature Importance"),
    code("import shap\nexplainer = shap.TreeExplainer(model)\nshap_vals = explainer.shap_values(X[:500])\nshap.summary_plot(shap_vals, pd.DataFrame(X[:500], columns=feats), show=False)\nplt.tight_layout()\nplt.show()"),
])

for name, nb in [("01_EDA", nb01), ("02_FIRMS_Analysis", nb02), ("03_Model_Analysis", nb03)]:
    p = Path(f"notebooks/{name}.ipynb")
    p.write_text(json.dumps(nb, indent=2))
    print(f"Written: {p}")

print("All notebooks created.")
