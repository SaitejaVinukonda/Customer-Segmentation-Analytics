import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
st.title("Customer Segmentation using K-Means Clustering")

@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

customer_data = load_data()

# Display data
st.subheader("Dataset Preview")
st.write(customer_data.head())

st.subheader("Basic Information")
st.write(customer_data.info())

st.subheader("Summary Statistics")
st.write(customer_data.describe())

# Visualization: Gender Distribution
st.subheader("Gender Distribution")
gender_counts = customer_data['Gender'].value_counts()
fig, ax = plt.subplots()
gender_counts.plot(kind='bar', color=['skyblue', 'pink'], ax=ax)
ax.set_title("Gender Comparison")
ax.set_xlabel("Gender")
ax.set_ylabel("Count")
st.pyplot(fig)

# Age Distribution
st.subheader("Age Distribution")
fig, ax = plt.subplots()
sns.histplot(customer_data['Age'], kde=True, bins=10, color='blue', ax=ax)
ax.set_xlabel("Age Class")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# Annual Income Distribution
st.subheader("Annual Income Distribution")
fig, ax = plt.subplots()
sns.histplot(customer_data['Annual Income (k$)'], kde=True, bins=10, color='purple', ax=ax)
st.pyplot(fig)

# Spending Score Distribution
st.subheader("Spending Score Distribution")
fig, ax = plt.subplots()
sns.histplot(customer_data['Spending Score (1-100)'], kde=True, bins=10, color='green', ax=ax)
st.pyplot(fig)

# K-means clustering
X = customer_data.iloc[:, [3, 4]].values

st.subheader("Optimal Cluster Selection using Elbow Method")
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

# User input for number of clusters
k = st.slider("Select the number of clusters:", min_value=2, max_value=10, value=6, step=1)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=50, random_state=125)
customer_data['Cluster'] = kmeans.fit_predict(X)

# PCA for visualization
scaler = StandardScaler()
scaled_features = scaler.fit_transform(X)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)
customer_data['PC1'] = principal_components[:, 0]
customer_data['PC2'] = principal_components[:, 1]

# Scatter plot for clusters
st.subheader("Cluster Visualization Using PCA")
fig, ax = plt.subplots()
sns.scatterplot(data=customer_data, x='PC1', y='PC2', hue='Cluster', palette='viridis', s=100, ax=ax)
ax.set_title("Clusters Visualization Using PCA")
st.pyplot(fig)

# Display cluster summary
st.subheader("Clustered Data Preview")
st.write(customer_data.head())
