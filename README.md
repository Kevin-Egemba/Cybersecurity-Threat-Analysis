# Cybersecurity Anomaly Detection & Threat Modeling

## Overview

This project explores how machine learning models can detect anomalous and malicious behavior across different cybersecurity environments.

The workflow is designed as a **full data science pipeline**, beginning with data validation and progressing through modeling and evaluation across multiple datasets.

Datasets used:

* **BETH** — system-level event logs (unsupervised anomaly detection)
* **UNSW-NB15** — labeled network intrusion dataset (supervised learning)
* **Cybersecurity Attacks dataset** — multiclass attack classification

---

## Project Workflow

### 1. Data Overview & Validation

Notebook: `01_data_overview_.ipynb`

Before building models, I perform:

* data quality checks
* validation of labels and detection of leakage risks
* distribution analysis (class imbalance, skew, outliers)
* feature sanity checks

This step ensures all downstream modeling is based on reliable and interpretable data.

---

### 2. Unsupervised Anomaly Detection (BETH)

Notebook: `02_beth_anomaly_detection.ipynb`

Approach:

* Train models on normal behavior distributions
* Generate anomaly scores using:

  * KMeans (distance-based scoring)
  * Gaussian Mixture Models (probabilistic scoring)
  * Isolation Forest (tree-based anomaly detection)
  * DBSCAN (density-based clustering)

Evaluation is performed using a proxy `sus` label to assess how well models rank suspicious events.

**Key Result:**
Isolation Forest and GMM provided the strongest separation between normal and suspicious behavior.

---

### 3. Supervised Intrusion Detection (UNSW-NB15)

Notebook: `03_unsw_supervised.ipynb`

Models:

* Logistic Regression
* Support Vector Machines (SVM)
* Random Forest
* Gradient Boosting

**Key Result:**
Tree-based models significantly outperformed linear models by capturing nonlinear attack patterns.

---

### 4. Attack Classification (Cyber Attacks Dataset)

Notebook: `04_cyber_attacks_supervised.ipynb`

Models:

* K-Nearest Neighbors (KNN)
* Support Vector Machines (SVM)
* Gradient Boosting

**Key Insight:**
Model performance was highly sensitive to preprocessing steps such as feature scaling and class imbalance handling.

---

### 5. Results Comparison

Notebook: `05_results_comparison.ipynb`

This notebook consolidates:

* performance across datasets
* comparison between unsupervised and supervised approaches
* final modeling tradeoffs and conclusions

---

## Project Structure

```bash id="3k9p0u"
network-security-capstone/
│
├── data/
│   ├── Beth DataSet/
│   ├── Cybersecurity Attacks DataSets/
│   └── Network Security DataSet/
│
├── raw/                  # raw input files (not version-controlled if large)
├── figures/              # saved visualizations
├── notebooks/
│   ├── 01_data_overview_.ipynb
│   ├── 02_beth_anomaly_detection.ipynb
│   ├── 03_unsw_supervised.ipynb
│   ├── 04_cyber_attacks_supervised.ipynb
│   └── 05_results_comparison.ipynb
│
├── results/
├── src/                  # future modular code (feature engineering, models)
├── README.md
├── requirements.txt
```

---

## Data Handling Note

Raw datasets are stored separately from processed data to maintain:

* reproducibility
* clarity between raw and transformed inputs
* flexibility for future pipeline development

Large files may be excluded from version control.

---

## Technologies Used

* Python (NumPy, Pandas)
* Scikit-learn
* Matplotlib / Seaborn
* Jupyter Notebooks

---

## Key Takeaways

* Unsupervised models (Isolation Forest, GMM) effectively detect anomalies without labeled data.
* Supervised models outperform in structured intrusion detection scenarios.
* Feature preprocessing and dataset characteristics strongly impact performance.
* Combining anomaly detection with classification provides a more robust detection strategy.

---

## Future Work

* Sequence-based modeling (LSTM / Transformers)
* Real-time anomaly detection pipelines
* Feature engineering on system-level arguments (`args`)
* Deployment using streaming frameworks (Kafka / Spark)

---

## Author

Kevin Egemba
M.S. Data Science — Boston University
