# 🛒 E-commerce Customer Segmentation & Prediction

> Unsupervised customer segmentation on UK online retail data using **RFM analysis** and **K-Means clustering**, followed by a supervised model to predict the segment of new customers.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)]()
[![Status](https://img.shields.io/badge/Status-Complete-success)]()

---

## 📌 Overview

E-commerce businesses lose revenue when they treat every customer the same. This project segments customers into actionable groups using behavioural data (recency of purchase, frequency, monetary value, and cancellation rate), then trains a classifier to assign **new** customers to the correct segment in real time.

**Dataset:** UK-based online retail transactions — invoice-level records of purchases, cancellations, and customer IDs. See *Dataset Setup* below for download instructions.

## 📥 Dataset Setup

This project uses the **Online Retail dataset** from the UCI Machine Learning Repository (~45 MB, not committed to this repo).

1. Download from: https://archive.ics.uci.edu/dataset/352/online+retail
2. Place the CSV file in the repo root and rename it to `data.csv`
3. Update the read path in the notebook if needed:
   ```python
   customer_data = pd.read_csv('data.csv', encoding='latin-1')
   ```

## 🎯 Approach

The pipeline runs in two stages:

**Stage 1 — Unsupervised segmentation**
1. **Data cleaning** — dropped nulls, removed zero-price/zero-quantity rows, separated cancelled orders
2. **Feature engineering (RFM + C)** — aggregated to customer level:
   - `DaysFreq` (Recency: days since last purchase)
   - `PurchaseFreq` (Frequency: number of invoices)
   - `TotalAmount` (Monetary: lifetime spend)
   - `CancellationsFreq` (% of orders cancelled)
3. **Scaling & outlier handling** — standardised features, separated outlier customers into their own group
4. **Clustering** — compared **K-Means**, **Agglomerative**, and **DBSCAN**; final choice: K-Means with `k=4` based on silhouette score
5. **Dimensionality reduction** — PCA for 2D/3D cluster visualisation

**Stage 2 — Supervised prediction**
6. **Decision Tree classifier** trained on labelled clusters → predicts segment for new customers without re-running clustering
7. **Cross-validation** for model robustness
8. **Model persistence** — final K-Means saved as `kmeansCustSegPrediction.pkl` for deployment

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `scikit-learn` · `Matplotlib` · `Seaborn` · `SciPy` · `Pickle`

**Algorithms used:** K-Means · Agglomerative Clustering · DBSCAN · PCA · Decision Tree

## 📊 Evaluation

- **Silhouette Score** — chosen k=4 based on highest score across tested values
- **Davies-Bouldin Index** — secondary cluster quality check
- **Inertia (WCSS)** — used for elbow method during k-selection
- **Cross-validation** — Decision Tree accuracy validated via k-fold

## 🚀 Quickstart

```bash
git clone https://github.com/Varun-Kumar-DS/ecommerce-customer-segmentation.git
cd ecommerce-customer-segmentation
pip install -r requirements.txt
jupyter notebook E_commerce_Customer_Segmentation_and_Prediction.ipynb
```

## 📁 Structure

```
├── E_commerce_Customer_Segmentation_and_Prediction.ipynb   # main notebook
├── kmeansCustSegPrediction.pkl                             # saved model
├── requirements.txt
└── README.md
```

## 📚 What I Learned

- **RFM is foundational but incomplete** — adding cancellation rate as a 4th dimension surfaced a previously-hidden problem-customer segment that pure RFM missed
- **K-Means assumes spherical clusters** — comparing against DBSCAN exposed where K-Means oversimplifies. Trade-off: K-Means is faster and easier to deploy, DBSCAN handles arbitrary shapes
- **Outliers shouldn't always be removed** — high-value customers often *look* like outliers but are the most important segment. Solution: separated them into their own labelled group
- **Why pair clustering with a classifier** — clustering alone can't scale to live data. Persisting cluster labels and training a Decision Tree on them gives you real-time segment assignment for new customers

## 📬 Contact

**Varun Kumar** · [Portfolio](https://varun-kumar-ds.github.io) · [LinkedIn](https://www.linkedin.com/in/varun-kumar-ai) · [varunzayne@gmail.com](mailto:varunzayne@gmail.com)
