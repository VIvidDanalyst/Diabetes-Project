![Diabetes](https://github.com/VIvidDanalyst/Diabetes-Project/assets/139154608/3da915c6-ac71-4e97-b47d-52dc77147af4)
---


## Diabetes-Project
**Diabetes** is a chronic metabolic disorder characterized by elevated levels of blood glucose, resulting from insufficient production or inefficient utilization of insulin. Insulin, a hormone produced by the pancreas, facilitates the absorption of glucose into cells for energy. In individuals with diabetes, this regulatory mechanism is impaired, leading to persistent hyperglycemia.

There are two main types of diabetes:

Type 1, often diagnosed in childhood, involves the immune system attacking and destroying insulin-producing cells
Type 2, more common in adults, is linked to lifestyle factors and insulin resistance.

If left unmanaged, diabetes can lead to serious complications, including cardiovascular diseases, kidney dysfunction, and nerve damage. Regular monitoring, lifestyle modifications, and, in some cases, medication or insulin therapy are crucial components of diabetes management.

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

 ### Data Cleaning/ Preparation
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






   
   
