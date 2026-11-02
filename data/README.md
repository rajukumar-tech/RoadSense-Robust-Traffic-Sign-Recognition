# GTSRB Dataset

This directory holds the German Traffic Sign Recognition Benchmark (GTSRB) dataset.
Per the project hygiene rules, large raw dataset files are not committed to git repository tracking.

### Dataset Sources:
1. **Official Source (INI / ERDA Mirror):**
   - Homepage: `https://benchmark.ini.rub.de/gtsrb_dataset.html`
   - Archive: `https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html`
   - Training Data: `GTSRB_Final_Training_Images.zip` (~276 MB)
   - Test Data: `GTSRB_Final_Test_Images.zip` (~88 MB)
   - Test Ground Truth: `GTSRB_Final_Test_GT.zip` (~100 KB)

2. **Kaggle Mirror:**
   - Link: `https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign`

### Automated Download & Extraction:
Run the automated download script:
```bash
python src/download_data.py
```
This script downloads and extracts training and test datasets into `data/GTSRB/`.
