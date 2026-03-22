# Cybersecurity Anomaly Detection & Threat Modeling

## Overview

This project applies machine learning techniques to detect anomalous and malicious activity across multiple cybersecurity datasets.

The project is structured as a **full analytical pipeline**, beginning with data validation and progressing through unsupervised and supervised modeling approaches.

---

## Project Workflow

### 1. Data Overview & Reality Checks

Notebook: `01_data_overview_.ipynb`

Before modeling, I validate key assumptions:

* Do datasets support the intended tasks?
* Are there risks of label leakage?
* What does "attack" look like numerically?
* What constraints (missingness, skew, imbalance) must be addressed?

This step ensures that all downstream modeling decisions are grounded in data reality.

---

### 2. Unsupervised Anomaly Detection (BETH)

Notebook: `beth_anomaly_detection_unsupervised.ipynb`

* Focus: system-level event data
* Models:

  * KMeans (distance-based scoring)
  * Gaussian Mixture Models (log-likelihood)
  * Isolation Forest
  * DBSCAN

Evaluation uses a proxy label (`sus`) to assess anomaly ranking.

---

### 3. Supervised Intrusion Detection (UNSW-NB15)

Notebook: `unsw_supervised_modeling.ipynb`

* Focus: labeled network traffic
* Models:

  * Logistic Regression
  * SVM
  * Random Forest
  * Gradient Boosting

---

### 4. Supervised Attack Classification (Cyber Attacks Dataset)

Notebook: `cyber_attacks_supervised_modeling.ipynb`

* Focus: multi-class attack classification
* Models:

  * KNN
  * SVM
  * Gradient Boosting

---

### 5. Model Comparison & Results

Notebook: `05_results_comparison.ipynb`

* Cross-dataset comparison
* Performance tradeoffs
* Model selection insights

---

## Key Results

* Isolation Forest showed strongest anomaly detection capability in unsupervised settings
* GMM provided the clearest probabilistic separation
* Tree-based models (Random Forest, Gradient Boosting) performed best on labeled datasets
* Model performance is highly sensitive to feature engineering and class imbalance

---

## Project Structure

```bash
network-security-capstone/
│
├── data/
│   ├── beth/
│   ├── unsw_nb15/
│   └── cyber_attacks/
│
├── notebooks/
│   ├── 01_data_overview_.ipynb
│   ├── beth_anomaly_detection_unsupervised.ipynb
│   ├── unsw_supervised_modeling.ipynb
│   ├── cyber_attacks_supervised_modeling.ipynb
│   └── 05_results_comparison.ipynb
│
├── figures/
├── results/
├── src/
├── README.md
└── requirements.txt
```

---

## Technologies Used

* Python (Pandas, NumPy)
* Scikit-learn
* Matplotlib / Seaborn
* Jupyter

---

## Key Takeaways

* Unsupervised methods can detect suspicious behavior without labels but require careful validation
* Tree-based models consistently outperform linear models in cybersecurity contexts
* Data preprocessing and feature engineering are critical for reliable detection systems

---

## Future Work

* Sequence modeling (LSTM / Transformers)
* Real-time detection pipelines
* Feature engineering on system-level arguments
* Hybrid models combining anomaly detection + classification

---

## Author

Kevin Egemba
Data Science | Cybersecurity | Analytics
