import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
file_path = r"D:\Downloads\New folder\Internship\titanic.csv"
data = pd.read_csv(file_path)
print(data.head(5))
print(data.info(8))

print("Missing values:\n", data.isnull().sum())

print("\nDuplicate rows:",data.duplicated().sum())

data = data.drop_duplicates()

print("\nDuplicate rows:", data.duplicated().sum())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Sex"] = le.fit_transform(data["Sex"])

data["Embarked"] = data["Embarked"].fillna("Unknown")
data["Embarked"] = le.fit_transform(data["Embarked"])

data["Age"] = data["Age"].fillna(data["Age"].median())

data = data.drop(["Name","Ticket","Cabin"], axis=1)

X = data.drop("Survived", axis=1)
y = data["Survived"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(max_iter=500)
log_reg.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
y_pred_log = log_reg.predict(X_test)

print("\nLogistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log, target_names=["Did not survive","Survived"]))

cm = confusion_matrix(y_test, y_pred_log)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Did not survive","Survived"])
disp.plot()
plt.title("Confusion Matrix - Logistic Regression")
plt.show()
