 # 🎯 Customer Segmentation Analysis using K-Means Clustering & Dash Board
This project aims to segment mall customers into distinct groups based on their demographic and spending behavior using **unsupervised machine learning**. The insights derived from this analysis help businesses target specific customer groups with tailored marketing strategies, thereby increasing customer satisfaction and revenue.
## 🧠 Objective
To perform **customer segmentation** using the **Mall Customer Dataset**, enabling a retail store to understand different types of customers based on attributes such as:
- Age
- Annual Income (k$)
- Spending Score (1–100)
By applying **K-Means Clustering**, we aim to group customers with similar characteristics together.
## 📊 Dataset Overview
The dataset (`Mall_Customers.csv`) typically includes the following features:
| CustomerID | Gender | Age | Annual Income (k$) | Spending Score (1–100) |
|------------|--------|-----|--------------------|-------------------------|
- **Age**: Age of the customer
- **Annual Income**: Estimated income in $000s
- **Spending Score**: Score assigned based on customer behavior and purchasing data (1 = low spender, 100 = high spender)
## 🧪 Project Workflow
1. **Data Cleaning & Preprocessing**:
   - Checking for null values, data types
   - Label encoding categorical features (e.g., Gender)
   - Feature selection for clustering (Age, Income, Spending Score)
2. **Exploratory Data Analysis (EDA)**:
   - Distribution plots for all features
   - Correlation heatmaps
   - Box plots by gender or age groups
3. **Feature Scaling**:
   - StandardScaler used to normalize feature space
4. **Elbow Method to find optimal k**:
   - Plotting Within-Cluster Sum of Squares (WCSS) against various k values
   - Identifying the 'elbow point' as the optimal number of clusters
5. **K-Means Clustering**:
   - Applying KMeans with chosen `k`
   - Assigning cluster labels to each customer
6. **Cluster Visualization**:
   - 2D scatter plots (e.g., Income vs Spending Score colored by cluster)
   - Optional 3D plot using Plotly
7. **Exporting Results**:
   - Final DataFrame with cluster labels saved to CSV for dashboarding
8. **Power BI Dashboard (Optional)**:
   - Upload clustered CSV to Power BI
   - Create interactive visuals: bar charts, filters by cluster, spending score comparison, etc
## 🔧 Technologies Used
- **Python 3**
- **Pandas**, **NumPy**
- **Matplotlib**, **Seaborn**
- **Scikit-learn** (for KMeans clustering and preprocessing)
- **Plotly** (optional for interactive plots)
- **Power BI** (for business dashboard)
## 🚀 How to Run the Code
1. Clone the repository:
git clone https://github.com/SaitejaVinukonda/customer-segmentation-Analytics
cd customer-segmentation-kmeans
2. Install required packages:
pip install -r requirements.txt
3. Run the script
python customer_segmentation.py
4. (Optional) Use the generated segmented_customers.csv in Power BI or Streamlit for visualization.
## 📈 Sample Output
- Cluster 0: Young high-income high-spending customers
- Cluster 1: Older customers with low income and low spending
- Cluster 2: High-income but conservative spenders
Visualizations help in interpreting these segments and their business value.
## 📂 File Structure
📁 customer-segmentation-kmeans
├── customer_segmentation.py      # Python code for analysis and clustering
├── Mall_Customers.csv            # Input dataset
├── segmented_customers.csv       # Output with cluster labels
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
## 📌 Key Learnings
- Practical application of **K-Means Clustering**
- Hands-on with **EDA & Feature Engineering**
- Importance of scaling and dimensionality in clustering
- Building a business-useful dashboard from ML output
- ☁️ Deployment on Streamlit Cloud
To deploy:
Create a GitHub repository
Push both health_streamlit_app.py and requirements.txt
Visit https://streamlit.io/cloud and link your GitHub repo
Streamlit will automatically install dependencies and run the app
## 📄 License
MIT License – feel free to use and modify this project.
## 🙋 Author
- **Your Name**
- GitHub:https://github.com/SaitejaVinukonda
- Here is the deployed link, you can watch my project lively:https://saitejavinukonda-customer-segmenta-customer-segmentation-m0cihk.streamlit.app/
