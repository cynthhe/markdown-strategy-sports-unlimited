import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data from an Excel file
df = pd.read_excel('data_for_MarkdownManagementAtSportsUnlimited.xlsx')

# Standardize column names to remove extra spaces
df.columns = df.columns.str.strip()

# Ensure required columns are available
required_columns = {'1st Ticket Price', '1st Markdown %', 'Lifecycle Length', 'Units Sales',
                    'Dollar Sales', 'Branded?', 'Unit Sales by Week 3', '1st Markdown in Week #'}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise KeyError(f"Missing columns in dataset: {missing_columns}")

# Fill missing markdown values with 0 (assuming no markdown if missing)
df['1st Markdown %'] = df['1st Markdown %'].fillna(0)

# Calculate Pre-Markdown Price (Before markdown was applied)
df['Pre_Markdown_Price'] = df['1st Ticket Price']

# Estimate Pre-Markdown Unit Sales
df['Pre_Markdown_Unit_Sales'] = np.where(
    (df['1st Markdown in Week #'].notna()) & (df['1st Markdown in Week #'] > 1),
    df['Unit Sales by Week 3'],
    np.nan  # If markdown happened in week 1, we don't have pre-markdown sales data
)

# Drop rows where pre-markdown sales are missing
df = df.dropna(subset=['Pre_Markdown_Unit_Sales'])

### CLUSTER ANALYSIS ###
features = df[['1st Ticket Price', '1st Markdown %', 'Lifecycle Length', 'Pre_Markdown_Unit_Sales', 'Dollar Sales', 'Branded?']]
features = features.fillna(features.mean())  # Fill remaining missing values
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Perform k-means clustering with k=4
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Store regression results
cluster_demand_models = {}

# Store revenue results
cluster_revenues = {}

# Suggested markdown policy based on STR levels for each cluster
markdown_policy = {
    0: {  # Cluster 0: Branded, Moderate Pricing, Longer Lifecycles
        "High": (0.10, 0.20),
        "Medium": (0.25, 0.30),
        "Low": (0.40, 0.50),
    },
    1: {  # Cluster 1: Lower-Priced, Shorter Lifecycles, High Unit Sales
        "High": (0.05, 0.15),
        "Medium": (0.20, 0.25),
        "Low": (0.30, 0.40),
    },
    2: {  # Cluster 2: Higher-Priced, Non-Branded, Short Lifecycles
        "High": (0.10, 0.20),
        "Medium": (0.25, 0.30),
        "Low": (0.40, 0.50),
    },
    3: {  # Cluster 3: High-Priced, Branded, Very Short Lifecycles
        "High": (0.05, 0.15),
        "Medium": (0.20, 0.25),
        "Low": (0.35, 0.40),
    }
}

# Set up the plot
plt.figure(figsize=(10, 6))

# Loop through each cluster and fit a demand function BEFORE markdown
for cluster in df['Cluster'].unique():
    cluster_data = df[df['Cluster'] == cluster]

    if len(cluster_data) < 2:  # Skip if not enough data points
        print(f"Skipping Cluster {cluster} due to insufficient data points.")
        continue

    X = cluster_data[['Pre_Markdown_Price']]
    X = sm.add_constant(X)  # Add intercept
    y = cluster_data['Pre_Markdown_Unit_Sales']

    # Perform regression analysis
    model = sm.OLS(y, X).fit()
    cluster_demand_models[cluster] = model

    # Extract coefficients
    a = round(model.params['const'], 2)
    b = round(model.params['Pre_Markdown_Price'], 2)

    # Print demand function formula
    print(f"Cluster {cluster} Demand Function: Q = {a} + {b}P")

    # Generate price range for demand curve plotting
    price_range = np.linspace(cluster_data['Pre_Markdown_Price'].min(), cluster_data['Pre_Markdown_Price'].max(), 100)
    demand_prediction = a + b * price_range

    # Plot demand curve
    plt.plot(price_range, demand_prediction, label=f'Cluster {cluster} (Q = {a} + {b}P)')
    plt.scatter(cluster_data['Pre_Markdown_Price'], cluster_data['Pre_Markdown_Unit_Sales'], alpha=0.6)

    # Estimate revenue under different markdown policies
    for str_level, markdown_range in markdown_policy[cluster].items():
        markdown_low, markdown_high = markdown_range
        for markdown in [markdown_low, markdown_high]:
            new_price = cluster_data['Pre_Markdown_Price'] * (1 - markdown)
            predicted_demand = a + b * new_price

            # Calculate revenue
            revenue = new_price * predicted_demand
            cluster_revenues[(cluster, str_level, markdown)] = revenue.mean()

# Final plot adjustments
plt.xlabel('Pre-Markdown Price')
plt.ylabel('Pre-Markdown Unit Sales')
plt.title('Demand Curves by Cluster')
plt.legend()
plt.grid()
plt.show()

# Output revenue calculations for each cluster and markdown strategy
for key, revenue in cluster_revenues.items():
    cluster, str_level, markdown = key
    print(f"Cluster {cluster}, STR Level: {str_level}, Markdown: {markdown*100}% => Estimated Revenue: ${revenue:,.2f}")
