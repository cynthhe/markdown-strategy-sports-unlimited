## 🏷️ Markdown Strategy: Sports Unlimited

### 📘 Overview

This project presents a data-driven approach to optimizing markdown strategies for Sports Unlimited, a retail chain with over 500 stores. The company’s current one-size-fits-all markdown policy relies on an aggressive 50% discount, leaving significant revenue on the table. The goal is to create a smarter, segmented markdown strategy based on item performance and product characteristics.

### 📊 Key Analyses
* Sell-Through Rate (STR) Analysis
  * Items grouped into High (>80%), Medium (50–80%), and Low (<50%) STR tiers
  * Identified markdown inefficiencies, especially for high/medium-performing items
* Clustering Analysis (Python)
  * Features: price, markdown %, lifecycle, sales, brand
  * Used the Elbow Method to identify 4 optimal clusters
  * Tailored markdown rules for each product cluster
* Regression (Excel)
  * Validated STR and cluster groupings as statistically significant
    
### ✅ Results & Recommendations
* Proposed segmented markdown strategy with tiered rules by cluster
* Aligned markdown timing and depth with product performance
* Projected +51% increase in sales compared to the current strategy

### 🚀 Tools & Technologies
* Python (Pandas, Scikit-learn, Matplotlib)
* Excel (Regression Analysis, Pivot Tables)

### 🔮 Future Improvements
* Real-time ML-based clustering
* Geo-specific markdowns using sales & weather data
* Integration of customer-level segmentation for personalized pricing
