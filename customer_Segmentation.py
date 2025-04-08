import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.title("🔍 General Purpose Clustering App using K-Means")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("📄 Dataset Preview")
    st.write(df.head())

    st.subheader("📋 Basic Info")
    st.write(df.dtypes)

    st.subheader("📊 Summary Statistics")
    st.write(df.describe())

    # Select features
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.sidebar.header("🧮 Feature Selection for Clustering")
    selected_features = st.sidebar.multiselect("Select numeric columns to use for clustering:", numeric_cols)

    if len(selected_features) >= 2:
        X = df[selected_features].dropna()

        # Elbow method
        st.subheader("📈 Optimal Clusters: Elbow Method")
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=123)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)

        fig, ax = plt.subplots()
        ax.plot(range(1, 11), wcss, marker='o')
        ax.set_title("Elbow Method")
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("WCSS")
        st.pyplot(fig)

        # Select number of clusters
        k = st.slider("Select number of clusters (K):", min_value=2, max_value=10, value=3, step=1)

        # Standardize + Cluster
        scaler = StandardScaler()
        scaled_X = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=50, random_state=123)
        cluster_labels = kmeans.fit_predict(scaled_X)
        df['Cluster'] = cluster_labels

        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_X)
        df['PC1'], df['PC2'] = pca_result[:, 0], pca_result[:, 1]

        # Cluster Plot
        st.subheader("🌀 Cluster Visualization (PCA)")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='tab10', s=80, ax=ax)
        ax.set_title("Clustered Data in 2D (via PCA)")
        st.pyplot(fig)

        st.subheader("📌 Clustered Data Preview")
        st.write(df.head())
    else:
        st.warning("⚠️ Please select at least 2 numeric columns for clustering.")
else:
    st.info("📁 Upload a CSV file to get started.")
