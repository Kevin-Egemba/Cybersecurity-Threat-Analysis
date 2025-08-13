# Network Security & Anomaly Detection

Overview

For my Data Science Capstone, I immersed myself in the world of network security, focusing on intrusion and anomaly detection. I worked with three distinct but complementary datasets:
	1.	BETH Dataset – Kernel-level system event logs with numeric features such as processId, userId, argsNum, and returnValue.
	2.	Cybersecurity Attacks Dataset – Network behavior data including Packet Length, Anomaly Scores, Severity Level, and labeled Attack Type.
	3.	UNSW-NB15 Dataset – Realistic network flow statistics including dur, spkts, dpkts, sbytes, rate, sload, and dload.

My goal was to investigate how both supervised and unsupervised machine learning methods could reveal structure in high-dimensional, noisy data and support real-world detection of cyber threats.

⸻

Technical Approach

I structured my project across Weeks 1–12, progressively building from preprocessing and EDA to advanced modeling. The key machine learning and clustering methods I applied were:

Supervised Learning

K-Nearest Neighbors (KNN)
	•	Explored both Euclidean and Manhattan distance metrics.
	•	Worked well on the lower-dimensional, balanced BETH dataset (Accuracy: 0.87, k=5), but struggled with the imbalanced Cybersecurity dataset (F1 Macro: 0.62).
	•	Insight: KNN excels in well-separated, balanced datasets but loses precision in overlapping or sparse spaces.

Gradient Boosting
	•	Implemented using sklearn.ensemble.GradientBoostingClassifier with GridSearchCV hyperparameter tuning (learning_rate, max_depth, n_estimators).
	•	Outperformed KNN in all datasets by capturing nonlinear feature interactions.
	•	BETH: AUC 0.91, Accuracy 0.93; UNSW: AUC 0.86.
	•	Key takeaway: Boosting handled class imbalance better and uncovered relationships between network traffic metrics.

⸻

Unsupervised Learning

K-Means Clustering
	•	Evaluated via Silhouette Scores and PCA visualization.
	•	Found optimal k=3 across datasets.
	•	BETH clusters aligned with high returnValue and sus scores; UNSW clusters reflected packet load and duration patterns.

DBSCAN (Density-Based Spatial Clustering)
	•	Effective at identifying dense anomaly regions (e.g., high returnValue events in BETH, high-throughput traffic in UNSW).
	•	Struggled with the flat density structure of the Cybersecurity dataset.
	•	Tuned eps and min_samples per dataset; best performance at eps=1.5 for BETH/UNSW.

Hierarchical Agglomerative Clustering (HAC)
	•	Explored ward and average linkage methods.
	•	Revealed nested substructures in BETH event logs and separated high-throughput vs. low-rate flows in UNSW.

⸻

Overfitting Prevention & Evaluation

To ensure model robustness:
	•	Cross-validation for KNN and Gradient Boosting.
	•	StandardScaler normalization for numeric features.
	•	GridSearchCV hyperparameter tuning.
	•	Silhouette Scores and visual inspection for clustering.
	•	Early stopping in Gradient Boosting to prevent overfitting.

⸻

Exploratory Data Analysis (EDA) Insights
	•	Extreme right skew in sload, rate, and Anomaly Scores.
	•	Strong correlations between sbytes and dload (>0.85 Pearson).
	•	Outlier detection via DBSCAN informed eps tuning.
	•	PCA dimensionality reduction improved cluster visualization.

⸻

Key Outcomes
	•	BETH Dataset: Both supervised (Gradient Boosting Precision: 0.90) and clustering methods excelled.
	•	Cybersecurity Dataset: Overlapping features made classification challenging; Boosting performed moderately well (F1: 0.71), DBSCAN flagged 30%+ as noise.
	•	UNSW-NB15 Dataset: Clear behavioral groupings in traffic flows; Boosting AUC: 0.86, DBSCAN clustered high-risk flows with Silhouette 0.27.

⸻

Next Steps

I plan to expand this work by:
	•	Implementing Isolation Forests for anomaly detection.
	•	Exploring deep clustering techniques to improve separation in noisy, overlapping datasets.
	•	Incorporating real-time stream processing for continuous intrusion detection.

⸻

Skills & Tools Demonstrated
	•	Languages/Frameworks: Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn.
	•	Techniques: Supervised & unsupervised ML, dimensionality reduction (PCA), hyperparameter tuning, cross-validation, clustering evaluation.
	•	Concepts: Anomaly detection, feature importance analysis, density-based clustering, overfitting mitigation, high-dimensional data visualization.

