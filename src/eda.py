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


plt.scatter(df['depth'], df['price'])
plt.xlabel("Depth")
plt.ylabel("Price")
plt.title("Depth vs Price")
plt.savefig("../outputs/depth_price.png")
plt.clf()
#Scatter plot shows that price increases with carat and for depth only a certain range of depth price increases and then decreases.


# Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.savefig("../outputs/heatmap.png")
#heatmap shows that carat has the highest correlation with price, followed by x, y, and z dimensions. Depth and table have low correlation with price.
print("EDA Completed")
