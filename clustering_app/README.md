🛒 Wholesale Customer Clustering – Flask + Machine Learning

This project is a web-based application that segments wholesale customers based on their annual spending using K-Means, Gaussian Mixture Model (GMM), and Hierarchical (Agglomerative) Clustering.

Users can enter customer spending values in a clean UI, click Predict, and instantly view cluster results.

✅ Features

✔ Web-based Machine Learning App using Flask
✔ Real-time clustering using K-Means & GMM
✔ Hierarchical (Agglomerative) Clustering included for analytical purposes
✔ Dataset is log-transformed & standardized before clustering
✔ Models saved and loaded using Joblib
✔ Modern UI with dark theme (HTML, CSS & JavaScript)
✔ Instant results (AJAX API → No page reload)

📁 Project Structure
clustering_app/
│
├── app.py                      # Flask backend (API + UI)
├── model_train.py              # Script to train models (K-Means, GMM, Hierarchical)
├── requirements.txt            # Required Python packages
│
├── models/                     # Saved models (.pkl files)
│   ├── scaler.pkl
│   ├── kmeans.pkl
│   ├── gmm.pkl
│   └── agg.pkl                 # Hierarchical model (used for analysis, not prediction)
│
├── templates/
│   └── index.html              # Web UI (Frontend)
│
├── dataset/
│   └── Wholesale customers data_clustering.csv
│
└── README.md

⚙️ Technology Stack
Layer	Technology
Backend API	Flask
Frontend	HTML, CSS, JavaScript
ML Models	K-Means, Gaussian Mixture, Agglomerative
Data Scaling	StandardScaler + log1p()
Model Saving	Joblib
Visualization (Optional)	Matplotlib (for dendrogram)
📊 Dataset Details

The dataset contains annual spending by wholesale customers across 6 categories:

Feature
Fresh
Milk
Grocery
Frozen
Detergents_Paper
Delicassen
🧠 Machine Learning Process

Load dataset (Wholesale customers data_clustering.csv)

Select required features

Apply log1p() transformation (fix skewness)

Standardize using StandardScaler

Train models:
✅ K-Means (3 clusters)
✅ Gaussian Mixture Model (3 clusters)
✅ Agglomerative Clustering (for analysis only)