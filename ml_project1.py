# 📌 2. Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# 📌 3. Load your dataset
df = pd.read_csv(r"C:\Users\VENKAT\OneDrive\Desktop\Sales.csv")

# 📌 4. Basic cleaning
for col in ['Models', 'Mobile']:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

df.dropna(subset=['Storage', 'Memory'], inplace=True)

# Convert 'Memory' and 'Storage' to numeric (e.g., '8 GB' -> 8)
df['Memory'] = df['Memory'].astype(str).str.extract(r'(\d+)').astype(float)
df['Storage'] = df['Storage'].astype(str).str.extract(r'(\d+)').astype(float)

# Encode categorical columns
label_encoders = {}
for col in ['Brands', 'Colors', 'Camera']:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = df[col].astype(str)  # Ensure all are strings
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# 📌 5. Define features and target
if 'Selling Price' not in df.columns:
    raise ValueError("Column 'Selling Price' not found in the dataset.")

X = df.drop(columns=['Selling Price'])
y = df['Selling Price']

# 📌 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 7. Train a model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 📌 8. Evaluate
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 📌 9. Save model and encoders
joblib.dump(model, 'selling_price_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
