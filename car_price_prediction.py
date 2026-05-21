# CodeAlpha Task 3: Car Price Prediction with ML
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# 1. LOAD DATA
df = pd.read_csv('dataset/car_data.csv')
print("="*60)
print("CODEALPHA TASK 3: CAR PRICE PREDICTION")
print("="*60)
print(f"Dataset Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst 5 rows:")
print(df.head())

# 2. DATA CLEANING
print("\n" + "="*60)
print("DATA CLEANING & FEATURE ENGINEERING")
print("="*60)
print("Missing values:\n", df.isnull().sum())
df = df.dropna()

# 3. FEATURE ENGINEERING - Car Age instead of Year
df['Car_Age'] = 2026 - df['Year'] 
df = df.drop('Year', axis=1)

# 4. ENCODE CATEGORICAL COLUMNS
le = LabelEncoder()
df['Car_Name'] = le.fit_transform(df['Car_Name'])
df['Fuel_Type'] = le.fit_transform(df['Fuel_Type'])
df['Selling_type'] = le.fit_transform(df['Selling_type'])
df['Transmission'] = le.fit_transform(df['Transmission'])

print("\nData after encoding:")
print(df.head())

# 5. SPLIT FEATURES & TARGET
X = df.drop('Selling_Price', axis=1) # All columns except target
y = df['Selling_Price'] # Target column

# 6. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set: {X_train.shape}, Testing set: {X_test.shape}")

# 7. TRAIN MODELS
print("\n" + "="*60)
print("MODEL TRAINING & EVALUATION")
print("="*60)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# 8. EVALUATE MODELS
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name}:")
    print(f" MAE : {mae:.2f} lakhs")
    print(f" RMSE : {rmse:.2f} lakhs")
    print(f" R2 Score : {r2:.4f}")
    return r2

lr_r2 = evaluate_model(y_test, lr_pred, "Linear Regression")
rf_r2 = evaluate_model(y_test, rf_pred, "Random Forest Regressor")

best_model = "Random Forest" if rf_r2 > lr_r2 else "Linear Regression"
print(f"\n🏆 Best Model: {best_model} with R2 Score: {max(lr_r2, rf_r2):.4f}")

# 9. PLOTS FOR CODEALPHA VIDEO SUBMISSION
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Actual vs Predicted
axes[0].scatter(y_test, rf_pred, alpha=0.6, color='royalblue', edgecolors='k')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Price (lakhs)', fontsize=12)
axes[0].set_ylabel('Predicted Price (lakhs)', fontsize=12)
axes[0].set_title('Random Forest: Actual vs Predicted', fontsize=14)

# Plot 2: Feature Importance
feature_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
feature_imp.tail(6).plot(kind='barh', ax=axes[1], color='forestgreen')
axes[1].set_title('Top 6 Important Features', fontsize=14)
axes[1].set_xlabel('Importance Score', fontsize=12)

# Plot 3: Model Comparison
models = ['Linear Regression', 'Random Forest']
scores = [lr_r2, rf_r2]
bars = axes[2].bar(models, scores, color=['orange', 'purple'])
axes[2].set_ylabel('R2 Score', fontsize=12)
axes[2].set_title('Model Performance Comparison', fontsize=14)
axes[2].set_ylim(0, 1)
for bar in bars:
    height = bar.get_height()
    axes[2].text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('car_price_prediction_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("PROJECT COMPLETE - READY FOR CODEALPHA SUBMISSION ✅")
print("Graph saved as 'car_price_prediction_results.png'")
print("="*60)