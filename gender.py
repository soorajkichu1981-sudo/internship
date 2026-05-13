import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path=r"D:\Downloads\New folder\assignment\gender_classification.csv"
df =pd.read_csv(file_path)
print(df.head(5))
print(df.info())
print("Missing values:\n", df.isnull().sum())
print("\nDuplicate rows:",df.duplicated().sum())
df = df.drop_duplicates()
print("\nDuplicate rows:", df.duplicated().sum())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["gender"] = le.fit_transform(df["gender"])
x = df.drop("gender", axis=1)
y = df["gender"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report

# Logistic Regression
y_pred_log = log_reg.predict(X_test)
print("\nLogistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log, target_names=["Female","Male"]))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred_log)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Female", "Male"])
disp.plot()
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

new_data = [[0, 13.5, 6.2, 1, 1, 0, 1]]  


new_data_df = pd.DataFrame(new_data, columns=x.columns)


log_result = log_reg.predict(new_data_df)[0]


print("Logistic Regression:", le.inverse_transform([log_result])[0])



