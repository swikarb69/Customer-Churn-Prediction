import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score

customer_df = pd.read_csv(r"data/Telco-Customer-Churn.csv")


# Type conversion
customer_df['TotalCharges'] = pd.to_numeric(customer_df['TotalCharges'], errors='coerce')

# Filling missing values
customer_df['TotalCharges'].fillna(customer_df['TotalCharges'].median())

# Target and Features
X = customer_df.drop(['customerID', 'Churn'], axis=1)
y = customer_df['Churn']

# Label Encoder
le = LabelEncoder()
y = le.fit_transform(y)

# One hot encoding dummies
X = pd.get_dummies(X)

# Train Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Model training
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)

# model = XGBClassifier(
#     n_estimators=300,
#     max_depth=6,
#     learning_rate=0.05,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42,
#     eval_metric='logloss'
# )

# model = DecisionTreeClassifier(
#     max_depth=5,
#     random_state=42
# )

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

print(
    feature_importance.sort_values(
        by='Importance',
        ascending=False
    ).head(10)
)

y_prob = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_prob)

print("ROC-AUC:", auc)

print("Model trained!")