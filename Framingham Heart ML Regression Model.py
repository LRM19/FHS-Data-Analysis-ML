import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, auc, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE 
import seaborn as sns

df = pd.read_csv(r'file_path') # Load dataframe

# DATA FILTERING

# Filling of the NULL values with KNN algorithm
df.education = KNNImputer(n_neighbors=5).fit_transform(df['education'].values.reshape(-1, 1))
df.glucose = KNNImputer(n_neighbors=5).fit_transform(df['glucose'].values.reshape(-1, 1))
df.BMI = KNNImputer(n_neighbors=5).fit_transform(df['BMI'].values.reshape(-1, 1))
df.totChol = KNNImputer(n_neighbors=5).fit_transform(df['totChol'].values.reshape(-1, 1))
df.BPMeds = KNNImputer(n_neighbors=5).fit_transform(df['BPMeds'].values.reshape(-1, 1))
df.cigsPerDay = KNNImputer(n_neighbors=5).fit_transform(df['cigsPerDay'].values.reshape(-1, 1))
df.heartRate = KNNImputer(n_neighbors=5).fit_transform(df['heartRate'].values.reshape(-1, 1))

# Since we don't have any categorical variable (labeled as such), but only integers ad floats, no data transformation is required

# ML Model Building

#Target Class count

plt.figure(figsize=(8,8))
plt.pie(df['TenYearCHD'].value_counts(), labels=['Negative','Positive'], autopct='%1.2f%%')
plt.show()

#Log Transform Continuous Variables

df_copy = df

df_copy['log_cigsPerDay'] = np.log1p(df_copy['cigsPerDay'])
df_copy['log_totChol'] = np.log1p(df_copy['totChol'])
df_copy['log_sysBP'] = np.log1p(df_copy['sysBP'])
df_copy['log_diaBP'] = np.log1p(df_copy['diaBP'])
df_copy['log_BMI'] = np.log1p(df_copy['BMI'])
df_copy['log_heartRate'] = np.log1p(df_copy['heartRate'])
df_copy['log_glucose'] = np.log1p(df_copy['glucose'])
df_copy['log_age'] = np.log1p(df_copy['age'])



#Normalizing dataset

scaler = StandardScaler()

cols = df_copy.drop(['TenYearCHD'], axis=1).columns

norm_df = scaler.fit_transform(df_copy.drop(['TenYearCHD'], axis=1))
norm_df = pd.DataFrame(data=norm_df, columns=cols, index=df_copy.drop(['TenYearCHD'], axis=1).index)                     

# Creating the train test spit 

x = norm_df
y = df_copy['TenYearCHD']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=23)
x_train.shape, x_test.shape, y_train.shape, y_test.shape

# train a logistic regression model on the training set

log_reg = LogisticRegression(solver='liblinear') # We use the liblinear solver since the datset we are working with is reasonably small
log_reg.fit(x_train, y_train)

log_pred = log_reg.predict(x_test)

# First Metrics Evaluation

print ('Accuracy Score :', accuracy_score(y_test, log_pred))
print ('Cross Validation Score : ', cross_val_score(log_reg, x_train, y_train, cv=5).mean())
print (classification_report(y_test, log_pred))

sns.heatmap(confusion_matrix(y_test, log_pred), annot=True)
plt.show()

#Accuracy Score : Accuracy Score in Imbalanced Dataset can be a Trap.

# We get illusion of High Accuracy because our Estimator Learns well from Majority Class and is able to Predict well 
# on Majority Class but not Minority Class leaving us in an illusion of High Accuracy. 
# 0.8603 seems like a good accuracy but it has no value since we can observe that we've Misclassification happening here for Minority Class.
# Cross-Validation Scores uses the average of the output, which will be affected by the number of folds. 
# Cross-Validation Scores Help us Identify if our Model is Over / Under-fitting.
# Classification Report : In Classification Report our Important metrics is Precision (TP/TP + FP) & Recall (TP/TP + FN). 
# We can see our Recall Scores is good only for Majority Class but Positive Class has Bad Recall Score.

# Confusion Matrix : Diagonal Values of Confusion Matrix are correct.
#  So we can see that In Negative Diagnosis out of 908 the 907 are correctly Classified while only 1 is misclassified,
#  we can also call it as Type 1 Error. While, In case of Positive Diagnosis out of 152 examples, 5 are Classified correctly 
# while rest 147 are misclassified as Negative Class.

# To try and improve the model we can implement Weight Class parameters

log_reg_cw = LogisticRegression(solver='liblinear', class_weight='balanced')
log_reg_cw.fit(x_train, y_train)

log_cw_pred = log_reg_cw.predict(x_test)

#Metrics Evaluation

print ('Accuracy Score :', accuracy_score(y_test, log_cw_pred))
print ('Cross Validation Score : ', cross_val_score(log_reg_cw, x_train, y_train, cv=5).mean())
print (classification_report(y_test, log_cw_pred))

sns.heatmap(confusion_matrix(y_test, log_cw_pred), annot=True)
plt.show()

# By adding Class Weight Parameter to our Estimator we've got good Recall Score.
# Its observable that for Negative Class 575 are Classified Correct & 333 are misclassified. 
# While for Positive Class 108 are Classified correct and 44 are misclassified, It is a better result than our previous prediction. 
# Type 2 Error has reduced upto some extent. 


# Implementing oversampling using SMOTE could help to get better results

#Applying SMOTE (createing synthetic observations based upon the existing minority observations)

smote = SMOTE(sampling_strategy='not majority')
x_s_res, y_s_res = smote.fit_resample(x_train, y_train)

est_reg = LogisticRegression(solver='liblinear', max_iter=1000, C=1).fit(x_s_res, y_s_res)
est_pred = est_reg.predict(x_test)

print ('Accuracy Score :', accuracy_score(y_test, est_pred))
print ('Cross Validation Score : ', cross_val_score(est_reg, x_s_res, y_s_res, cv=5).mean())
print (classification_report(y_test, est_pred))

sns.heatmap(confusion_matrix(y_test, est_pred), annot=True)
plt.show()

# From Confusion Matrix above, Its observable that for Negative Class 581 are Classified Correct & 327 are misclassified. 
# While for Positive Class 108 are Classified correct and 44 are misclassified, It is a better result than our previous prediction 
# in case of Type 1 Error has reduced upto some extent. Type 2 Errors remains similar.

# Evaluating the model using ROC AUC indicator


log_prob_cw = log_reg_cw.predict_proba(x_test)
log_prob_up = est_reg.predict_proba(x_test)

fpr_cw, tpr_cw, _ = roc_curve(y_test, log_prob_cw[:,1])
fpr_up, tpr_up, _ = roc_curve(y_test, log_prob_up[:,1])

log_cw_roc_auc = auc(fpr_cw, tpr_cw)
log_up_roc_auc = auc(fpr_up, tpr_up)

plt.figure(figsize=(10,7))

plt.plot(fpr_cw, tpr_cw, color=(np.random.rand(), np.random.rand(), np.random.rand()), label='AUC (Log. Reg.) = %0.4f'% log_cw_roc_auc)
plt.plot(fpr_up, tpr_up, color=(np.random.rand(), np.random.rand(), np.random.rand()), label='AUC (Log. Reg. (Post Upsamp.)) = %0.4f'% log_up_roc_auc)

plt.plot([0,1], 'grey', lw=2, linestyle='-.')

plt.legend(loc='lower right')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC-Area Under the Curve')
plt.show()

# The more the ROC-AUC Score the better is the model.
# We can evaluate Models performance based on this. It's clear that Logistic Regression model with Class Weight as "balanced" is giving us decent 
# Score of 0.7320 followed by Logistic Regression post Over-Sampling which has given us Score of 0.7213.
