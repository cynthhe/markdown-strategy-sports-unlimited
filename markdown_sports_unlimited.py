from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from an Excel file
df = pd.read_excel('data_for_MarkdownManagementAtSportsUnlimited.xlsx')

### ELBOW CURVE ###
# Select relevant features for clustering
features = df[['Branded?', '1st Ticket Price', '1st Markdown %', 'Lifecycle Length', 'Units Sales', 'Dollar Sales']]

# Handle missing values by imputing with the mean
features = features.fillna(features.mean())

# Standardize the features for clustering
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Determine the optimal number of clusters using the elbow method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_) # sum of squared distances of samples to their closest cluster center (Lower inertia values indicate that the data points are closer to their assigned cluster centers.)

# Plot the elbow curve
plt.figure(figsize=(9, 6))
plt.plot(range(1, 11), inertia, marker='o', color='#766CDB')
plt.title('Elbow Method for Optimal Clusters', fontsize=20, pad=15, weight='semibold', color='#222222')
plt.xlabel('Number of Clusters', fontsize=16, labelpad=10, weight='medium', color='#333333')
plt.ylabel('Inertia', fontsize=16, labelpad=10, weight='medium', color='#333333')
plt.xticks(fontsize=14, color='#555555')
plt.yticks(fontsize=14, color='#555555')
plt.grid(axis='both', linestyle='--', alpha=0.7)
plt.show()

print("Elbow curve plotted. Choose the optimal number of clusters based on the curve.")

### CLUSTER ANALYSIS ###

# Selected features for clustering
features = df[['1st Ticket Price', '1st Markdown %', 'Lifecycle Length', 'Units Sales', 'Dollar Sales', 'Branded?']]

# Imput missing values with the mean
features = features.fillna(features.mean())

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Perform k-means clustering with k=4 (optimal clusters from elbow analysis)
kmeans = KMeans(n_clusters=4, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_features)

# Attach cluster labels to the original dataframe
df['Cluster'] = cluster_labels

# Analyze the clusters: summary stats on each cluster
cluster_summary = df.groupby('Cluster')[['Branded?']].agg(['mean', 'median', 'min', 'max', 'std'])
# '1st Ticket Price', '1st Markdown %', 'Lifecycle Length', 'Units Sales', 'Dollar Sales', 'Branded?'
print('Cluster Summary Statistics with Branded Feature:')
print(cluster_summary)

# Visualize clusters using pairplot to explore relationships between features colored by cluster
sns.pairplot(df, vars=['1st Ticket Price', '1st Markdown %', 'Lifecycle Length', 'Units Sales', 'Dollar Sales'], hue='Cluster', palette='Set2')
plt.suptitle('Pairplot of Clusters with Branded Feature', fontsize=20, y=1.02)
plt.show()

# Boxplot for Lifecycle Length by Cluster
plt.figure(figsize=(9, 6))
sns.boxplot(x='Cluster', y='Lifecycle Length', data=df, palette='viridis', width=0.6, showfliers=True)

# Add titles and labels
plt.title('Boxplot: Lifecycle Length by Cluster', fontsize=20, pad=15, weight='semibold', color='#222222')
plt.xlabel('Cluster', fontsize=16, labelpad=10, weight='medium', color='#333333')
plt.ylabel('Lifecycle Length (weeks)', fontsize=16, labelpad=10, weight='medium', color='#333333')
plt.xticks(fontsize=14, color='#555555')
plt.yticks(fontsize=14, color='#555555')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

print('Boxplot for Lifecycle Length by Cluster created successfully.')

### Getting IDs within each Cluster ###
# Create a dictionary to store the IDs of each cluster
cluster_ids = {}

# Group by the 'Cluster' column and get the list of IDs for each cluster
for cluster_num in df['Cluster'].unique():
    cluster_ids[cluster_num] = df[df['Cluster'] == cluster_num]['ID'].tolist()

# Print the IDs for each cluster
for cluster_num, ids in cluster_ids.items():
    print(f"Cluster {cluster_num} contains the following IDs: {ids}")

# Create a new DataFrame with 'ID' and 'Cluster' columns
id_cluster_df = df[['ID', 'Cluster']]

# Export the DataFrame to an Excel file (you can change the file path as needed)
id_cluster_df.to_excel('clustered_ids.xlsx', index=False)

# Print the first few rows to verify
print(id_cluster_df.head())