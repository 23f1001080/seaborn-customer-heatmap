# chart.py
# Generates a professional Seaborn correlation heatmap for customer engagement
# Author: Sakshi (contact: 23f1001080@ds.study.iitm.ac.in)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Create realistic synthetic customer engagement data ---
# We'll simulate 8 engagement metrics across 400 customers/sessions
np.random.seed(42)
n = 400
data = {
    "Session_Length_min": np.clip(np.random.normal(15, 6, n), 1, None),
    "Pages_Visited": np.clip(np.random.poisson(4, n) + np.random.normal(0,1,n), 1, None),
    "Clicks": np.clip(np.random.poisson(8, n) + np.random.normal(0,2,n), 0, None),
    "Add_to_Cart": np.random.binomial(1, 0.18, n) + np.random.binomial(1, 0.05, n),
    "Purchase_Amount_USD": np.clip(np.random.exponential(60, n) + np.random.normal(0,20,n), 0, None),
    "Return_Visits_30d": np.clip(np.random.poisson(1, n), 0, None),
    "Email_Opens_30d": np.clip(np.random.poisson(2, n), 0, None),
    "Support_Contacts": np.random.poisson(0.2, n)
}

df = pd.DataFrame(data)

# Add some controlled correlations for realism
df["Pages_Visited"] += (df["Session_Length_min"] / 10).round(0)
df["Clicks"] += (df["Pages_Visited"] * 0.5).round(0)
df["Purchase_Amount_USD"] += df["Add_to_Cart"] * np.random.normal(60, 20, n)

# --- Compute correlation matrix ---
corr = df.corr()

# --- Plot styling ---
sns.set_style("white")
sns.set_context("talk", font_scale=0.9)

plt.figure(figsize=(8, 8))   # 8x8 inches at dpi=64 -> 512x512 px
cmap = sns.diverging_palette(220, 20, as_cmap=True)  # blue-red diverging

ax = sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap=cmap,
    vmin=-1,
    vmax=1,
    center=0,
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"shrink": 0.7, "label": "Pearson r"}
)

ax.set_title("Customer Engagement Metrics — Correlation Matrix", pad=16, fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

# Save exactly 512x512 pixels:
plt.savefig("chart.png", dpi=64, bbox_inches="tight")  # 8in * 64dpi = 512px
plt.close()

print("chart.png written (512x512 px).")
