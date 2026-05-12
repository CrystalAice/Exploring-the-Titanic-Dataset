#importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt  
import seaborn as sns

#read the file
try:
    train_data = pd.read_csv('train.csv')
except FileNotFoundError:
    print('File is not found.')
except Exception as e:
    print(f'Loading failed because of {e} error.')


#finding the data types, shape and any missing values
print(f'The data types in this dataset is as follows:\n{train_data.dtypes}')
print(f'The rows to column ratio of the data is as (rows, columns):\n{train_data.shape}')

columns = [
    'PassengerId', 'Pclass', 'Name', 'Sex', 
    'Age', 'SibSp', 'Parch', 'Ticket', 
    'Fare', 'Cabin', 'Embarked', 'Survived'

]

print('\nMissing values per column are as follows:') 
for col in columns:
   print(f"{col}: {train_data[col].isnull().sum()}")

#fill NULL values in the age column with 0 to be able to perform survival analysis
train_data['Age'] = train_data['Age'].fillna(0) 
train_data['Age'] = train_data['Age'].astype(int) #changing age data type from floats to integers

for col in columns: #fill null values in other columns
    train_data[col] = train_data[col].fillna('Unknown')

#Analysis of survival rates by gender, class, and age group
survival_Bygender = train_data.groupby('Sex')['Survived'].sum()
survival_Byage = train_data.groupby('Age')['Survived'].sum()
survival_Byclass = train_data.groupby('Pclass')['Survived'].sum()

print(f'\nThe survival rate by gender is:\n{survival_Bygender}')
print(f'\nThe survival rate by age is:\n{survival_Byage}')
print(f'\nThe survival rate by class is:\n{survival_Byclass}')

#creating bar plots and histograms for survival by demographics
#survival barplot by passenger class
sns.set_theme(style = 'whitegrid')
plt.figure(figsize = (12, 8))
sns.barplot(x='Pclass', y='Survived', data=train_data, palette = 'dark', hue = 'Pclass')
plt.xlabel("Pclass", fontsize = 24)
plt.ylabel("Survived", fontsize = 24)
plt.title("Survival Rate by Passenger Class", fontsize = 24)
plt.savefig('Survival Rate by Pclass.png')

#survival barplot by gender
sns.barplot(x='Sex', y='Survived', data=train_data, palette = 'dark', hue = 'Sex')
plt.xlabel("Pclass", fontsize = 24)
plt.ylabel("Survived", fontsize = 24)
plt.title("Survival Rate by Gender", fontsize = 24)
plt.savefig('Survival Rate by gender.png')

#create a histogram of the age distribution
new_train_data = train_data[train_data["Age"] != 0]
sns.histplot(new_train_data['Age'], bins = 20, kde = True)
plt.title('Distribution of age', fontsize = 22)
plt.xlabel('Age', fontsize = 22)
plt.ylabel('Number per age group', fontsize = 20)
plt.savefig('Age distribution.png')

#create a crosstab for age groups
bins = [0, 12, 19, 35, 60, 100]
labels = ['Child(s)', 'Teens', 'Young Adults', 'Adults', 'Seniors']
train_data['AgeGroup'] = pd.cut(train_data['Age'], bins = bins, labels = labels)
survived_byAGE = pd.crosstab(train_data['AgeGroup'], train_data['Survived'], normalize = 'index')
survived_byAGE.plot(kind = 'bar', stacked = True)
plt.title("Survival Distribution by Age Group", fontsize = 20)
plt.ylabel("Survival Frequency", fontsize = 18)
plt.xlabel('Age Groups', fontsize = 18)
plt.xticks(rotation = 45)
plt.savefig('Survival by age groups.png')
