import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("../data/diamonds.csv")
os.makedirs("../outputs", exist_ok=True)
# Scatter plot
plt.scatter(df['carat'], df['price'])
plt.xlabel("Carat")
plt.ylabel("Price")
plt.title("Carat vs Price")
plt.savefig("../outputs/carat_price.png")
plt.clf()

# Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.savefig("../outputs/heatmap.png")

print("EDA Completed")
