#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


# In[2]:


df= pd.read_csv(r"C:\Users\ibori\Downloads\diabetes_binary_health_indicators_BRFSS2015.csv")
## Data importation


# In[3]:


df


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df.hist(figsize=(25,20))
plt.show()


# In[7]:


df["Diabetes_binary"] = df["Diabetes_binary"].astype(int)
df.Diabetes_binary.hist(figsize=(8,7))
plt.show()


# ##### Change the float data type to int data type for easy computation by the model 

# In[8]:


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


# In[9]:


# Check correlation using a heatmap
plt.figure(figsize = (30,20))
sns.set(font_scale=1.5)
sns.heatmap(df.corr(numeric_only=True),annot=True, cmap='Blues')
plt.title("Diabetes Variable Correlations",fontsize=30)


# In[10]:


corr = df.corrwith(df['Diabetes_binary'])


# In[11]:


corr = pd.DataFrame(corr)


# In[12]:


corr


# In[13]:


plt.figure(figsize=(3,7))
sns.heatmap(corr, cmap='Blues', annot=True, fmt='.2%')


# # Noteable Correlations:
# 
# 
# #### Highest correlation : highBP, highchol, highBMI, genhealth, physhealth, diffwalk

# In[124]:


labels = ('Non_Daibetic','Diabetic')
plt.pie(df.Diabetes_binary.value_counts(),labels = labels,autopct='%.2f'  )
plt.title('Target Value Count')
plt.show()


# In[15]:


df.Diabetes_binary.value_counts()


# ###### 86.07% (218,334) of the patients do not have diabetes
# ###### 13.93% (35,346) of the patients have diabetes

# ##### Since this are the columns with high correlation with the target varaible i will be performing EDA on the corrlated columns 
# ###### Highest correlation : highBP, highchol, highBMI, genhealth, physhealth, diffwalk

# In[16]:


df.HighBP.value_counts()


# In[17]:


df['diabetes_binary_str']= df['Diabetes_binary'].replace({0:'Non Diabetic',1:'Diabetic'})



# In[18]:


df['HighBP_str']= df['HighBP'].replace({0:'No ',1:'Yes'})


# In[19]:


pd.crosstab(df.HighBP_str,df.diabetes_binary_str).plot(kind='bar',figsize=(8,8) )
plt.title('High Blood Pressure vs Diabetes Frequency')
plt.xlabel('HighBP')
plt.ylabel('Frequency')
plt.show()


# In[20]:


(df.groupby('diabetes_binary_str')['HighBP_str'].value_counts()/df.groupby('diabetes_binary_str')['HighBP_str'].count()).round(3)*100


# In[21]:


df['Highchol_str']= df['HighChol'].replace({0:'No ',1:'Yes'})


# In[22]:


pd.crosstab(df.Highchol_str,df.diabetes_binary_str).plot(kind='bar',figsize=(8,8) )
plt.title('High chol vs Diabetes Frequency')
plt.xlabel('Highchol')
plt.ylabel('Frequency')
plt.show()


# In[23]:


(df.groupby('diabetes_binary_str')['Highchol_str'].value_counts()/df.groupby('diabetes_binary_str')
 ['Highchol_str'].count()).round(4)*100

 
# In[24]:


(df.groupby(["HighBP_str", "Highchol_str"])["diabetes_binary_str"].value_counts()/df.groupby(["HighBP_str" , "Highchol_str"])
 ["Diabetes_binary"].count()).round(4)*100


# In[25]:


df['diffwalk_str']= df['DiffWalk'].replace({0:'No ',1:'Yes'})


# In[26]:


pd.crosstab(df.diffwalk_str,df.diabetes_binary_str).plot(kind='bar',figsize=(8,8) )
plt.title('Diffwalk vs Diabetes Frequency')
plt.xlabel('Diffwalk')
plt.ylabel('Frequency')
plt.show()


# In[27]:


(df.groupby('diabetes_binary_str')['diffwalk_str'].value_counts()/df.groupby('diabetes_binary_str')
 ['diffwalk_str'].count()).round(4)*100


# In[29]:


(df.groupby('diabetes_binary_str')['GenHlth'].value_counts()/df.groupby('diabetes_binary_str')
 ['GenHlth'].count()).round(4)*100


# In[30]:


df_no = df[df['Diabetes_binary'] == 0]
df_yes = df[df['Diabetes_binary'] == 1]
df_no_genhlth = df_no['GenHlth']
df_yes_genhlth = df_yes['GenHlth']


# In[113]:


sns.kdeplot(df_no_genhlth,color='purple')
sns.kdeplot(df_yes_genhlth,color='orange')
plt.grid()
plt.title('General Heath vs Diabetes_Binary Distribution')
plt.legend(['Not Diabetic', 'Diabetic'])
plt.show()


# In[34]:


df_no_income= df_no['Income']
df_yes_income = df_yes['Income']


# In[120]:


sns.kdeplot(df_no_income,color='green')
sns.kdeplot(df_yes_income,color='yellow')
plt.grid()
plt.title('Income vs Diabetes_Binary Distribution')
plt.legend(['Not Diabetic', 'Diabetic'])
plt.show()


# In[40]:


df_no_edu= df_no['Education']
df_yes_edu = df_yes['Education']


# In[115]:


sns.kdeplot(df_no_edu,color='green')
sns.kdeplot(df_yes_edu,color='blue')
plt.grid()
plt.title('Education vs Diabetes_Binary Distribution')
plt.legend(['Not Diabetic', 'Diabetic'])
plt.show()


# In[42]:


df_no_physhlth = df_no['PhysHlth']
df_yes_physhlth = df_yes['PhysHlth']


# In[121]:


sns.kdeplot(df_no_physhlth,color='red')
sns.kdeplot(df_yes_physhlth,color='orange')
plt.grid()
plt.title('General Heath vs Diabetes_Binary Distribution')
plt.legend(['Not Diabetic', 'Diabetic'])
plt.show()


# In[45]:


df_no_age = df_no['Age']
df_yes_age = df_yes['Age']


# In[122]:


sns.kdeplot(df_no_age,color='red')
sns.kdeplot(df_yes_age,color='blue')
plt.grid()
plt.title('Age vs Diabetes_Binary Distribution')
plt.legend(['Not Diabetic', 'Diabetic'])
plt.show()


# In[47]:


df.info()


# ### Feature Engineering 

# In[31]:


#### remove the string columns created 


# In[32]:


df.info()


# In[48]:


dropc= ['diabetes_binary_str','HighBP_str','diffwalk_str']


# In[49]:


df.drop(dropc,axis=1,inplace=True)


# In[51]:


df.drop('Highchol_str',axis=1,inplace=True)


# In[52]:


df.info()


# In[175]:


X =  df.iloc[:,1:]


# In[176]:


y = df.iloc[0:,:1]


# In[177]:


y.shape


# In[178]:


X.shape


# In[179]:


from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.feature_selection import chi2


# In[180]:


fsel = SelectKBest(score_func=f_classif, k=10)


# In[181]:


X_selected = fsel.fit_transform(X,y)


# In[182]:


X_selected.shape


# In[183]:


pd.DataFrame(X_selected)


# In[210]:


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

#f_Scores.sort_values(by=['Score'],ascending=False)


# In[211]:


X.shape


# In[212]:


plt.figure(figsize=(10, 6))
sns.barplot(x='Score', y='Feature', data=top_features)
plt.title('Chi-Square Feature Selection')
plt.xticks()
plt.show()


# In[71]:


del_colomns = ['Fruits', 'Veggies', 'Sex', 'CholCheck', 'AnyHealthcare','Education','Smoker','NoDocbcCost']


# In[72]:


model_df= df.drop(columns=del_colomns,axis=1)


# In[73]:


model_df


# In[74]:


y = model_df['Diabetes_binary']
X= model_df.drop('Diabetes_binary',axis=1)


# In[75]:


X.shape


# In[76]:


y.shape


# In[77]:


X


# In[76]:


from sklearn.model_selection import train_test_split


# In[79]:


nm = NearMiss(version = 1,n_neighbors=5)
x_sm,y_sm=nm.fit_resample(X,y)


# In[81]:


y_sm.value_counts()


# In[89]:


X_train,X_test,y_train,y_test = train_test_split(x_sm,y_sm,test_size=.20,random_state=30)


# In[90]:


for x in [X_train, X_test, y_train, y_test]:
    print(len(x))


# In[98]:


np.reshape(y_sm,(-1,1))


# In[103]:


y_sm


# In[102]:


y


# In[214]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


# In[125]:


import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")



# In[120]:


grid_models = [(KNeighborsClassifier(),[{'n_neighbors':[8,10,13],'weights':['uniform','distance'],'algorithm':['auto','ball_tree','brute']}]),
               (DecisionTreeClassifier(),[{'criterion':['gini','entropy','log_loss'],'min_samples_leaf':[4,5,6],'max_depth':[8,10,13]}]), 
               (RandomForestClassifier(),[{'n_estimators':[50,100,150,200],'max_depth':[8,10,13],'criterion':['gini','entropy'],'max_features':[1,3,5]}]),
               (LogisticRegression(),[{'C':[1.0, 0.80, 0.70]}]),
               (XGBClassifier(), [{'learning_rate': [0.01,0.03,0.05], 'min_child_weight':[1,3,5], 'eval_metric':['error','auc']}])]


# In[127]:


for i,j in grid_models:
    grid = GridSearchCV(estimator=i,param_grid = j, scoring = 'accuracy',cv=2)
    grid.fit(X_train, y_train)
    best_accuracy = grid.best_score_
    best_param = grid.best_params_
    print('{}:\nBest Accuracy : {:.2f}%'.format(i,best_accuracy*100))
    print('Best Parameters : ',best_param)
    print('')
    print('----------------')
    print('')
  


# ### A 
# #### KNeighborsClassifier()

# In[129]:


kn_classifier = KNeighborsClassifier(n_neighbors=13,algorithm='auto',weights='uniform') 


# In[130]:


kn_classifier.fit(X_train,y_train)


# In[131]:


kn_pred= kn_classifier.predict(X_test)


# In[133]:


from sklearn.metrics import *


# In[136]:


accuracy = accuracy_score(y_test,kn_pred )
classification_rep = classification_report(y_test, kn_pred)

# Print the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_rep)


# ### B
# #### Decisiontreeclassifier()

# In[139]:


dec = DecisionTreeClassifier(criterion='log_loss',max_depth=13,min_samples_leaf=5)


# In[140]:


dec.fit(X_train,y_train)


# In[141]:


y_pred= dec.predict(X_test)


# In[143]:


accuracy = accuracy_score(y_test,y_pred )
classification_rep = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_rep)


# # C 
# #### RandomForest 

# In[145]:


rf = RandomForestClassifier( criterion='entropy', n_estimators =100, max_depth=13,max_features=3)
rf.fit(X_train, y_train)


# In[146]:


y_predrf = rf.predict(X_test)


# In[148]:


accuracy = accuracy_score(y_test,y_predrf )
classification_rep = classification_report(y_test, y_predrf)

# Print the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_rep)


# ### D
# #### LogisticRegression()

# In[149]:


lg= LogisticRegression(C=0.7)
lg.fit(X_train,y_train)


# In[150]:


lgpred= lg.predict(X_test)


# In[151]:


accuracy = accuracy_score(y_test,lgpred )
classification_rep = classification_report(y_test, lgpred)

# Print the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_rep)


# ### E
# #### Xgbclassifier

# In[154]:


xg =XGBClassifier(eval_metric= 'error', learning_rate= 0.05, min_child_weight= 1)
xg.fit(X_train,y_train)


# In[155]:


xgpred = xg.predict(X_test)


# In[156]:


accuracy = accuracy_score(y_test,xgpred )
classification_rep = classification_report(y_test, xgpred)

# Print the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_rep)


# In[ ]:




