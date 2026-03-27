from ucimlrepo import fetch_ucirepo 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
  
# fetch dataset 
abalone = fetch_ucirepo(id=1) 
  
# data (as pandas dataframes) 
X = abalone.data.features 
y = abalone.data.targets 

print("Type of X: ", type(X))
print("Type of y: ", type(y))
  
"""
# metadata 
print(abalone.metadata) 
  
# variable information 
print(abalone.variables) 
"""

# visualize class distribution
"""
print("class distribution:\n", y.value_counts())

y.value_counts().plot(kind='bar')
plt.title("Abalone rings class distribution")
plt.show()
"""
"""
print(X.head())
print(y.head())
"""

X = pd.get_dummies(X, columns=['Sex'], dtype=int)

y_prepared = y.copy()

# binning 1 and 2 Rings together
y_prepared['Rings'] = y_prepared['Rings'].apply(lambda x: 2 if x <= 2 else x)

# binning 24 and above together
y_prepared['Rings'] = y_prepared['Rings'].apply(lambda x: 24 if x >= 24 else x)

print(y_prepared.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y_prepared, test_size=0.2, random_state=42, stratify=y_prepared)


# --- Comparison Study: "BEFORE" (Imbalanced) ---
print("Running Baseline Model (Before Balancing)...")
model_before = RandomForestClassifier(random_state=42)
model_before.fit(X_train, y_train.values.ravel()) # using values on y_train returns 2d array, and then use ravel() to return 1d array
y_pred_before = model_before.predict(X_test)

# --- SMOTE
print("Applying SMOTE and Training Balanced Model (After)...")
sm = SMOTE(random_state=42, k_neighbors=1)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

model_after = RandomForestClassifier(random_state=42)
model_after.fit(X_train_res, y_train_res.values.ravel())
y_pred_after = model_after.predict(X_test)

# --- 5. Final Results and Comparison ---
print("Results before(Imbalanced)")
print(classification_report(y_test, y_pred_before, zero_division=0))

print("Results after(Balanced)")
print(classification_report(y_test, y_pred_after, zero_division=0))

# --- 6. Visualization of the change
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

y_train.value_counts().sort_index().plot(kind='bar', ax=ax1, color='skyblue')
ax1.set_title("Original (binned) distribution")

y_train_res.value_counts().sort_index().plot(kind='bar', ax=ax2, color='salmon')
ax2.set_title("SMOTE (binned) resampled distribution")

plt.tight_layout()
plt.show()