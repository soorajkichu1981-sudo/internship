import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
file_path =r"D:\Downloads\New folder\Internship\titanic.csv"
df = pd.read_csv("titanic.csv")
print(df.head())
print(df.shape)
print(df.info())
#counting the duplicate values
print(df.duplicated().value_counts())
print(df.isnull().sum())
missing = df.isnull().sum()
# Create the bar plot
ax = missing.plot(kind='bar', color='skyblue')
plt.title('Missing Values per Column')
plt.ylabel('Count of Missing Values')
plt.xlabel('Column')
plt.xticks(rotation=45)

# ✨ Add text labels above each bar
for index, value in enumerate(missing):
    plt.text(index, value + 5, str(value), ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()
df = df.drop(columns=['Cabin'])  # Cabin has 687 missing — too many!
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Count the number of people who survived and didn't
survival_counts = df['Survived'].value_counts()

print("Survival Counts:\n", survival_counts)

sns.countplot(x='Survived', data=df, palette='pastel')
plt.title('Survival Count (0 = Died, 1 = Survived)')
plt.xlabel('Survived')
plt.ylabel('Passenger Count')
plt.show()
# Count of passengers by gender
gender_counts = df['Sex'].value_counts()
print("\nGender Counts:\n", gender_counts)
# Count of survived grouped by gender
survival_by_gender = df.groupby(['Sex', 'Survived']).size()
print("\nSurvival by Gender:\n", survival_by_gender)

sns.countplot(x='Sex', hue='Survived', data=df, palette='Set2')
plt.title('Survival by Gender')
plt.xlabel('Gender')
plt.ylabel('Passenger Count')
plt.legend(title='Survived')
plt.show()

# Count of passengers by class
pclass_counts = df['Pclass'].value_counts()
print("\nPassenger Class Counts:\n", pclass_counts)

# Labels for the pie chart
labels = ['Class 1 (Upper)', 'Class 2 (Middle)', 'Class 3 (Lower)']

# Plotting the pie chart
plt.figure(figsize=(7, 7))
plt.pie(pclass_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=['lightblue', 'seashell', 'olive'])
plt.title('Passenger Class Distribution (Pclass)', fontsize=14)
plt.axis('equal')  # Ensures it's a circle
plt.show()

sns.kdeplot(df['Age'], fill=True, color='skyblue')
plt.title('Age Distribution (KDE Plot)')
plt.xlabel('Age')
plt.ylabel('Density')
plt.show()

df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 40, 60, 100], labels=['Child', 'Teen', 'Adult', 'Middle Age', 'Senior'])
sns.countplot(x='AgeGroup', hue='Survived', data=df)
plt.title('Survival by Age Group')
plt.show()


