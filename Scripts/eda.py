# eda.py — Full EDA + Preprocessing Pipeline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Set working directory
os.chdir("C:/Gokul Important things/Content Monetization Modeler")

# Load dataset
df = pd.read_csv("data/youtube_ad_revenue_dataset.csv")  # Make sure this file exists in /data

# Create output folders
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/cleaned", exist_ok=True)

# EDA: Basic info
print("🔍 Dataset Info:")
print(df.info())
print("\n📊 Summary Stats:")
print(df.describe())

# Target distribution
sns.histplot(df['ad_revenue_usd'], bins=50, kde=True)
plt.title("Distribution of Ad Revenue")
plt.savefig("outputs/plots/ad_revenue_distribution.png")
plt.clf()

# Numeric scatter plots
numeric_cols = ['views', 'likes', 'comments', 'watch_time_minutes', 'video_length_minutes', 'subscribers']
for col in numeric_cols:
    sns.scatterplot(x=df[col], y=df['ad_revenue_usd'])
    plt.title(f"{col} vs Ad Revenue")
    plt.savefig(f"outputs/plots/{col}_vs_revenue.png")
    plt.clf()

# Correlation matrix
corr_matrix = df[numeric_cols + ['ad_revenue_usd']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.savefig("outputs/plots/correlation_matrix.png")
plt.clf()

# Categorical boxplots
cat_cols = ['category', 'device', 'country']
for col in cat_cols:
    sns.boxplot(x=df[col], y=df['ad_revenue_usd'])
    plt.title(f"Ad Revenue by {col}")
    plt.xticks(rotation=45)
    plt.savefig(f"outputs/plots/revenue_by_{col}.png")
    plt.clf()

# Missing values
missing_percent = df.isnull().sum() / len(df) * 100
print("\n🧹 Missing Values (%):")
print(missing_percent)

# Duplicates
duplicate_count = df.duplicated().sum()
print(f"\n🧹 Duplicate Rows: {duplicate_count}")

# Drop duplicates
df = df.drop_duplicates()

# Fill missing values
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Log transform target
df['log_revenue'] = df['ad_revenue_usd'].apply(lambda x: np.log1p(x))

# Preprocessing: Encoding + Scaling
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first'), cat_cols)
    ]
)

X = df[numeric_cols + cat_cols]
X_transformed = preprocessor.fit_transform(X)

# Save cleaned dataset
X_df = pd.DataFrame(X_transformed.toarray() if hasattr(X_transformed, 'toarray') else X_transformed)
X_df['log_revenue'] = df['log_revenue']
X_df.to_csv("outputs/cleaned/cleaned_dataset.csv", index=False)

print("\n✅ EDA + Preprocessing complete. Cleaned dataset saved.")
