import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("../data/diamonds.csv", index_col=0)

df = df.drop_duplicates().dropna()

# Feature engineering
df['volume'] = df['x'] * df['y'] * df['z']

# Remove invalid values
df = df[(df['x'] > 0) & (df['y'] > 0) & (df['z'] > 0)]

# Encoding
le = LabelEncoder()
for col in ['cut', 'color', 'clarity']:
    df[col] = le.fit_transform(df[col])

# Scaling
scaler = StandardScaler()
X = df.drop('price', axis=1)
X_scaled = scaler.fit_transform(X)

X = pd.DataFrame(X_scaled, columns=X.columns)
y = df[['price']]

# Save
X.to_csv("../data/X_processed.csv", index=False)
y.to_csv("../data/y.csv", index=False)

print("Preprocessing Done")