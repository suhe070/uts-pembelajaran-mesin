# =========================
# IMPORT LIBRARY
# =========================
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv('citrus.csv')

print("=== HEAD DATA ===")
print(df.head())

print("\n=== INFO DATA ===")
print(df.info())

print("\n=== DESCRIBE DATA ===")
print(df.describe())

# =========================
# CEK MISSING VALUE
# =========================
print("\n=== MISSING VALUE ===")
print(df.isnull().sum())

# Drop missing value (kalau ada)
df = df.dropna()

print("\n=== SETELAH DROP MISSING VALUE ===")
print(df.isnull().sum())

# =========================
# ENCODING LABEL
# =========================
le = LabelEncoder()
df['name'] = le.fit_transform(df['name'])

print("\n=== HASIL ENCODING ===")
print(df.head())

# =========================
# FITUR & TARGET
# =========================
X = df.drop('name', axis=1)
y = df['name']

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# TRAINING MODEL
# =========================

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# SVM
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# =========================
# EVALUASI AKURASI
# =========================
acc_dt = accuracy_score(y_test, y_pred_dt)
acc_nb = accuracy_score(y_test, y_pred_nb)
acc_svm = accuracy_score(y_test, y_pred_svm)

print("\n=== AKURASI ===")
print("Decision Tree:", acc_dt)
print("Naive Bayes:", acc_nb)
print("SVM:", acc_svm)

# =========================
# CONFUSION MATRIX
# =========================

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_dt)).plot()
plt.title("Confusion Matrix - Decision Tree")
plt.show()

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_nb)).plot()
plt.title("Confusion Matrix - Naive Bayes")
plt.show()

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_svm)).plot()
plt.title("Confusion Matrix - SVM")
plt.show()

# =========================
# GRAFIK AKURASI
# =========================
models = ['Decision Tree', 'Naive Bayes', 'SVM']
accuracy = [acc_dt, acc_nb, acc_svm]

plt.figure()
plt.bar(models, accuracy)
plt.title('Perbandingan Akurasi Model')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.show()

# =========================
# FEATURE IMPORTANCE (BONUS)
# =========================
importance = dt.feature_importances_
features = df.drop('name', axis=1).columns

plt.figure()
plt.barh(features, importance)
plt.title('Feature Importance - Decision Tree')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
