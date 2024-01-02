![Diabetes](https://github.com/VIvidDanalyst/Diabetes-Project/assets/139154608/3da915c6-ac71-4e97-b47d-52dc77147af4)
---


## Diabetes-Project
**Diabetes** is a chronic metabolic disorder characterized by elevated levels of blood glucose, resulting from insufficient production or inefficient utilization of insulin. Insulin, a hormone produced by the pancreas, facilitates the absorption of glucose into cells for energy. In individuals with diabetes, this regulatory mechanism is impaired, leading to persistent hyperglycemia.

There are two main types of diabetes:

Type 1, often diagnosed in childhood, involves the immune system attacking and destroying insulin-producing cells
Type 2, more common in adults, is linked to lifestyle factors and insulin resistance.

If left unmanaged, diabetes can lead to serious complications, including cardiovascular diseases, kidney dysfunction, and nerve damage. Regular monitoring, lifestyle modifications, and, in some cases, medication or insulin therapy are crucial components of diabetes management.

### TABLE OF CONTENT
- [Project Overview](#project_overview)
- [Data Source](#data_source)
- [Tools](#tools)
- [Data Cleaning / Preperation](#data_cleaning_/_preparation)

### Project Overview
---
This data science project aims to leverage advanced analytics and machine learning teckniqies to gain valueble insights to diabetics related trends, risk factors and predictive modelling.
 The Primary **Objective** of this project is to develop predictive models that can enhance our understanding of diabetics onset progressin and complications. By Harnessing the power of data science, we aim to identify key factors influencing *Daibetes* and provide actionable insights for healthcare proffesionals and individual at risk.  

 This project works is to test 5 different machine learning algorithms to predict diabetes among 21 variables.

 ### Data Source 
 ---
 The primary dataset used for this project is the *Cdc's BRFSS2015 data* The behavioral risk factor surveillance (BRFSS) is an health related telephone survey administered by the *CDC* gathering responses from over 400,000 Americans. The survey conducted since 1984 delves into various aspect of including health-related risk behaviors, chronic health conditions, and the utilization of preventative services.  For the purposes of this project, data from the 2015 survey was utilized, obtained in CSV format from *Kaggle*  .This dataset has 253,680 survey responses to the CDC's BRFSS2015 survey, and  21 feature variables. These variables consist of either directly posed questions to participants or derived values calculated from individual responses.  

 ### Tools !🧰
 ---
 - PYTHON JUPYTAL NOTEBOOK

 ### Data Cleaning / Preparation
 ---
1.  Remove duplicate :No
   - I choose not to remove duplicate because these duplicates could show patterns and if removed could hinder the accuracy of the models
2.  Remove Outliers : No
   - Most of the variables are Categorical yes/no the dataset is formated to prevent outliers
3.   Remove Null values : No
   - There appears to be no null values in the dataset 
4.   Change Data Types : Yes
   - Changing all floats data types to ints to ensure computational speed


 ### Data Cleaning & Exploratory Data Analysis (EDA)
 ---
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
## Libraries Importation

df= pd.read_csv(r"C:\Users\ibori\Downloads\diabetes_binary_health_indicators_BRFSS2015.csv")
## Data importation

df.isnull().sum()
## checking for Null values

df.info()
## checking data information(datatypes, count of records) 

df.hist(figsize=(25,20))
plt.show()

# Change all floats to ints
df["Diabetes_binary"] = df["Diabetes_binary"].astype(int)
df["HighBP"] = df["HighBP"].astype(int)
df["HighChol"] = df["HighChol"].astype(int)
df["CholCheck"] = df["CholCheck"].astype(int)
df["BMI"] = df["BMI"].astype(int)
df["Smoker"] = df["Smoker"].astype(int)
df["Stroke"] = df["Stroke"].astype(int)
df["HeartDiseaseorAttack"] = df["HeartDiseaseorAttack"].astype(int)
df["PhysActivity"] = df["PhysActivity"].astype(int)
df["Fruits"] = df["Fruits"].astype(int) 
df["Veggies"] = df["Veggies"].astype(int)
df["HvyAlcoholConsump"] = df["HvyAlcoholConsump"].astype(int)
df["AnyHealthcare"] = df["AnyHealthcare"].astype(int)
df["NoDocbcCost"] = df["NoDocbcCost"].astype(int)
df["GenHlth"] = df["GenHlth"].astype(int)
df["MentHlth"] = df["MentHlth"].astype(int)
df["PhysHlth"] = df["PhysHlth"].astype(int)
df["DiffWalk"] = df["DiffWalk"].astype(int)
df["Sex"] = df["Sex"].astype(int)
df["Age"] = df["Age"].astype(int)
df["Education"] = df["Education"].astype(int)
df["Income"] =df["Income"].astype(int)
```

![Diabetes](https://github.com/VIvidDanalyst/Diabetes-Project/assets/139154608/b1da2641-3794-4b6a-8017-3f88d5ad308e)

```python
##Check correlation using a heatmap
plt.figure(figsize = (30,20))
sns.set(font_scale=1.5)
sns.heatmap(df.corr(numeric_only=True),annot=True, cmap='Blues')
plt.title("Diabetes Variable Correlations",fontsize=30)

corr = df.corrwith(df['Diabetes_binary']) ### correlating the target varible with the independent variables 
corr = pd.DataFrame(corr)
plt.figure(figsize=(3,7))
sns.heatmap(corr, cmap='Blues', annot=True, fmt='.2%')
```
![corr](https://github.com/VIvidDanalyst/Diabetes-Project/assets/139154608/82659a5f-9dd8-47e9-9901-c810e3c19977)
![corrwith](https://github.com/VIvidDanalyst/Diabetes-Project/assets/139154608/4ccd5df8-ab44-46a6-a023-b40a5475a2e3)

## Key Piont 
---
From the heatmap analysis above there are several independent variable that influence the dependent variable (diabetes_binary) possitively or negative. The highest positively correlated to the dependent variable is the Gen_Hlth variable, Income variable is the highest negatively correlated to the dependent variable with -16.39%.
- Noticeble variables 
   1. Gen_Hlth(+)
   2. High_BP(+)
   3. Diff_walk(+)
   4. BMI(+)
   5. High chol(+)
   6. Age(+)
   7. Heartdiseaseorattack(+)
   8. Income(-)
   9. Education(-)

```Python
df.Diabetes_binary.value_counts()

labels = ('Non_Daibetic','Diabetic')
plt.pie(df.Diabetes_binary.value_counts(),labels = labels,autopct='%.2f'  )
plt.show() #Visaulization of the Target Variable 

#Creating string variablles with the Binary datatype (0/1)
df['diabetes_binary_str']= df['Diabetes_binary'].replace({0:'Non Diabetic',1:'Diabetic'})
df['HighBP_str']= df['HighBP'].replace({0:'No ',1:'Yes'})
df['Highchol_str']= df['HighChol'].replace({0:'No ',1:'Yes'})
df['diffwalk_str']= df['DiffWalk'].replace({0:'No ',1:'Yes'})


##Viz
pd.crosstab(df.HighBP_str,df.diabetes_binary_str).plot(kind='bar',figsize=(8,8) )
plt.title('High Blood Pressure vs Diabetes Frequency')
plt.xlabel('HighBP')
plt.ylabel('Frequency')
plt.show()

pd.crosstab(df.Highchol_str,df.diabetes_binary_str).plot(kind='bar',figsize=(8,8) )
plt.title('High chol vs Diabetes Frequency')
plt.xlabel('Highchol')
plt.ylabel('Frequency')
plt.show()

pd.crosstab(df.diffwalk_str,df.diabetes_binary_str).plot(kind='bar',figsize=(8,8) )
plt.title('Diffwalk vs Diabetes Frequency')
plt.xlabel('Diffwalk')
plt.ylabel('Frequency')
plt.show()

pd.crosstab(df.GenHlth,df.diabetes_binary_str).plot(kind='bar',figsize=(8,8) )
plt.title('GenHlth vs Diabetes Frequency')
plt.xlabel('GenHlth')
plt.ylabel('Frequency')
plt.show()

## splitting data for visualization 
df_no = df[df['Diabetes_binary'] == 0]
df_yes = df[df['Diabetes_binary'] == 1]

df_no_genhlth = df_no['GenHlth']
df_yes_genhlth = df_yes['GenHlth']
sns.kdeplot(df_no_genhlth,color='purple')
sns.kdeplot(df_yes_genhlth,color='orange')
plt.grid()
plt.title('General Heath vs Diabetes_Binary Distribution')
plt.legend(['Not Diabetic', 'Diabetic'])
plt.show()

df_no_income= df_no['Income']
df_yes_income = df_yes['Income']
sns.kdeplot(df_no_income,color='green')
sns.kdeplot(df_yes_income,color='yellow')
plt.grid()
plt.title('Income vs Diabetes_Binary Distribution')
plt.legend(['Not Diabetic', 'Diabetic'])
plt.show()

df_no_edu= df_no['Education']
df_yes_edu = df_yes['Education']
sns.kdeplot(df_no_edu,color='green')
sns.kdeplot(df_yes_edu,color='blue')
plt.grid()
plt.title('Education vs Diabetes_Binary Distribution')
plt.legend(['Not Diabetic', 'Diabetic'])
plt.show()

df_no_physhlth = df_no['PhysHlth']
df_yes_physhlth = df_yes['PhysHlth']
sns.kdeplot(df_no_physhlth,color='red')
sns.kdeplot(df_yes_physhlth,color='orange')
plt.grid()
plt.title('General Heath vs Diabetes_Binary Distribution')
plt.legend(['Not Diabetic', 'Diabetic'])
plt.show()

```
## EDA VIZ
---
![EDAviz](https://github.com/VIvidDanalyst/Diabetes-Project/assets/139154608/cd3b1d0b-1e1d-47b0-bfc4-24414ee9175b)

---

![EDAviz](https://github.com/VIvidDanalyst/Diabetes-Project/assets/139154608/bc09b17b-ecd7-40c0-a3b7-bbb5b2d65832)

---

### Findings From EDA Performed
The EDA performed on the datasets highlighted several keypiont, some of the independent varaibles are (positively or negatively) Correlated to the target varaible. 

- Positively Correlated
    - The Variables positively correlated with the target variables are  Gen_Hlth,High_BP,Diff_walk,BMI,High chol,Age, Heartdiseaseorattack. What this connote is that when there is increase or the binary is yes the likelyhood of the patient being diabetic is high. For instance from the EDA viz the higher the age bracket the higher the chances of being diabetic and also if a patient is having difficulty walking the higher the chances of being diabetic vice-versa.
- Negatively Correlated
     - Independent variables like Income and Education is the highest correlated (nagatively) with the dependent varaibles. For instance the higher the income of individual the lower chances of being diabetic and also the Educational level of individuals enables healthy eating hence preventing diabetes. when the income is low the chances of being diabetic is high.
 
 ### Feature Engineering & Selection 
- I removed the created string variables created for EDA
  
  - Feature selection is carried out to improve model perfomance and also ensure fast training of models by reducing the independent variables.
``` python
##importing needed libraries for feature selection 
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.feature_selection import chi2

k=13
bestftt = SelectKBest(score_func=chi2)
fit = bestftt.fit(X,y)
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X.columns)
#concatenating two dataframes for better visualization
f_Scores = pd.concat([df_columns,df_scores],axis=1)
f_Scores.columns = ['Feature','Score']
n=f_Scores.shape[0]
print(n)
top_features = f_Scores.sort_values(by='Score', ascending=False).iloc[:k]
top_features
### this syntax choose the top 13 features chi2 score
del_colomns = ['Fruits', 'Veggies', 'Sex', 'CholCheck', 'AnyHealthcare','Education','Smoker','NoDocbcCost'] # drop the rest variables
model_df= df.drop(columns=del_colomns,axis=1)
```
### Balancing the Data
- The original data with over 200,000 records is inbalance. To ensure high performance of all models we will be balancing the the data to create equal reprensentation of all outcome in the target variable.
  ```python
  nm = NearMiss(version = 1,n_neighbors=5)
  x_sm,y_sm=nm.fit_resample(X,y) ##using nearmiss to Balance the data 
  ```
### Construcction of Models
 In this project work i will be using five(5) Models to choose the model with high accuracy score they are
 - K Nearest Neighbors
- Decision Tree
- Random Forests
- Logistic Regression
- XGBoost

### Model comparism
Comparing the accuracy of different models is crucial, especially in fields like medical diagnosis where identifying positive cases (e.g., detecting cancer) is of utmost importance. In such domains, minimizing the chance of missing positive cases (avoiding false negatives) is a priority due to the potentially severe consequences.

Upon reviewing the models, it is evident that there is a notable risk of Type 2 errors, where false negatives occur. Recall, which is the ability to capture positive instances, and Type 2 error are inversely related (1 - recall = Type 2 error). This is particularly worrisome in medical scenarios, as a negative diagnosis when a condition is actually positive can have life-altering implications.

Instead of solely focusing on accuracy, it is imperative to consider both accuracy and Type 2 error. Currently, XGBoost stands out with the lowest rate of Type 2 errors and the highest accuracy. The emphasis will now be on adjusting these models to enhance recall, ensuring a more robust ability to detect positive cases and minimizing the risk of false negatives.







  















   
   
