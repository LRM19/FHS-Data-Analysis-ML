import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.impute import KNNImputer
import seaborn as sns

df = pd.read_csv(r'file_path') # Load dataframe

#The dataset is a rather small subset of the FHS dataset, having 4240 observations and 16 variables. The variables are as follows:
# sex : the gender of the observations. binary var named “male”
# age : Age at the time of medical examination in years
# education : Categorical variable of the participants education, with the levels: Some high school (1), high school/GED (2), some college/vocational school (3), college (4)
# currentSmoker: Current cigarette smoking at the time of examinations
# cigsPerDay: Number of cigarettes smoked each day
# BPmeds: Use of Anti-hypertensive medication at exam
# prevalentStroke: Prevalent Stroke (0 = free of disease)
# prevalentHyp: Prevalent Hypertensive. Subject was defined as hypertensive if treated
# diabetes: Diabetic according to criteria of first exam treated
# totChol: Total cholesterol (mg/dL)
# sysBP: Systolic Blood Pressure (mmHg)
# diaBP: Diastolic blood pressure (mmHg)
# BMI: Body Mass Index, weight (kg)/height (m)^2
# heartRate: Heart rate (beats/minute)
# glucose: Blood glucose level (mg/dL)
# TenYearCHD : Target variable. The 10 year risk of coronary heart disease(CHD).

# Visualizing the dataset features

print(df.head())
print ('Number of Records (ROWS):', df.shape[0], '\nNumber of Features (COLUMNS): ', df.shape[1])
print(df.info())

# DATA FILTERING

# Percentages of NULL values for each feature:
print(df.isnull().sum()/df.shape[0]*100)

# Filling of the NULL values with KNN algorithm
df.education = KNNImputer(n_neighbors=5).fit_transform(df['education'].values.reshape(-1, 1))
df.glucose = KNNImputer(n_neighbors=5).fit_transform(df['glucose'].values.reshape(-1, 1))
df.BMI = KNNImputer(n_neighbors=5).fit_transform(df['BMI'].values.reshape(-1, 1))
df.totChol = KNNImputer(n_neighbors=5).fit_transform(df['totChol'].values.reshape(-1, 1))
df.BPMeds = KNNImputer(n_neighbors=5).fit_transform(df['BPMeds'].values.reshape(-1, 1))
df.cigsPerDay = KNNImputer(n_neighbors=5).fit_transform(df['cigsPerDay'].values.reshape(-1, 1))
df.heartRate = KNNImputer(n_neighbors=5).fit_transform(df['heartRate'].values.reshape(-1, 1))

# Alternatively, we can plot the Pearson Correlation Coefficient between the different features of the dataset and fill in the NULL values based on the existing relationships

# Pearson Correlation Coefficient visualization
# sns.heatmap(df.corr(), annot=True)
# plt.show()

#def impute_median(data):
#    return data.fillna(data.median())
#median imputation

# df.glucose = df['glucose'].transform(impute_median)
# df.education = df['education'].transform(impute_median)
# df.heartRate = df['heartRate'].transform(impute_median)
# df.totChol = df['totChol'].transform(impute_median)
# df.BPMeds = df['BPMeds'].transform(impute_median)

# group by classes that are in relation with other classes

# by_currentSmoker = df.groupby(['currentSmoker'])
# df.cigsPerDay = by_currentSmoker['cigsPerDay'].transform(impute_median)

# by_age = df.groupby(['male','age'])
# df.BMI = by_age['BMI'].transform(impute_median)

# Checking NULL values after the filling process
print(df.isnull().sum())

# Since we don't have any categorical variable (labeled as such), but only integers ad floats, no data transformation is required

# EXPLORATORY DATA ANALYSIS (EDA)

# To have an idea of the values' ranges and nature (Dicrete/Continuous) we use
print(df.describe())
# Values which have only 0 or 1 as values are our "categorical" variables (discrete), so let us analyse the continuous ones

# Distribution of values

#Distribution of Continuous variables
plt.figure()
plt.subplots_adjust(wspace=0.3, hspace=0.3)

plt.subplot(2, 3, 1)
sns.histplot(df['glucose'],kde=True)
plt.title('Distribution of Glucose')

plt.subplot(2, 3, 2)
sns.histplot(df['totChol'],kde=True)
plt.title('Distribution of Total Cholesterol')

plt.subplot(2, 3, 3)
sns.histplot(df['sysBP'],kde=True)
plt.title('Distribution of Systolic BP')

plt.subplot(2, 3, 4)
sns.histplot(df['diaBP'],kde=True)
plt.title('Distribution of Diastolic BP')

plt.subplot(2, 3, 5)
sns.histplot(df['BMI'],kde=True)
plt.title('Distribution of BMI')

plt.subplot(2, 3, 6)
sns.histplot(df['heartRate'],kde=True)
plt.title('Distribution of HeartRate') 

plt.show()

# We create a list of the continuous variables in the dataset and iterate through it to test the normality of the data

continuous = ['glucose','totChol','sysBP','diaBP','BMI','heartRate']

for feature in continuous:
    statistic, p_value = sp.stats.shapiro(df[feature])
    
    alpha = 0.05  # Significance level
    print(f"Shapiro-Wilk Test:")
    print(f"Statistic: {statistic:.5f}, p-value: {p_value}")
    
    if p_value > alpha:
        print("Sample looks Gaussian (fail to reject H0)")
    else:
        print("Sample does not look Gaussian (reject H0)")

# Now we proceed to analyse the age and CHD distribution in the patiens

sns.histplot(df['age'], kde=True)
plt.ylabel('Count')
plt.title('Age distribution')
plt.show()

sns.catplot(x='male', hue='TenYearCHD', data=df, kind='count',legend=False)
plt.xlabel('Gender')
plt.xticks(ticks=[0,1], labels=['Female', 'Male'])
plt.ylabel('Count')
plt.legend(['Negative', 'Positive'])
plt.title('CHD by Gender')
plt.show()

# To perform a more in depth analysis of the set, we can split (encode) the age and heartRate variables in groups
def encode_age(data):
    if data <= 40:
        return '32-40'
    if data > 40 and data <=55:
        return '40-60'
    else:
        return '>60'    

def heartrate_enc(data):
    if data <= 60:
        return 'Low'
    if data > 60 and data <=100:
        return 'Normal'
    else:
        return 'High'

df['enc_hr'] = df['heartRate'].apply(lambda x: heartrate_enc(x))
df['encode_age'] = df['age'].apply(lambda x: encode_age(x))

plt.subplots_adjust(wspace=0.3, hspace=0.3)

plt.subplot(1, 2, 1)
sns.countplot(df['enc_hr'], order=['Low', 'Normal', 'High'])
plt.xlabel('Heart Rate Categories')
plt.title('HeartRate (Grouped)')

plt.subplot(1, 2, 2)
sns.countplot(df['encode_age'], order=['32-40', '40-60', '>60'])
plt.xlabel('Age Groups')
plt.title('Count by Age Group')
plt.show()

# We can use the groups we defined to perform a grouped analysis of the other variables in the dataset

# Systolic and Diastolic Blood Pressure
plt.subplots_adjust(wspace=0.2, hspace=0.3)

plt.subplot(1, 2, 1)
sns.boxplot(x='encode_age', y='sysBP', hue='male', data=df, palette='pastel', order=['32-40', '40-60', '>60'])
plt.xlabel('Age Group / Gender')
plt.ylabel('Systolic BP')
plt.title('Systolic BP by Age Group & Gender')
plt.legend(title='Gender')

plt.subplot(1, 2, 2)
sns.boxplot(x='encode_age', y='diaBP', hue='male', data=df, palette='pastel', order=['32-40', '40-60', '>60'])
plt.xlabel('Age Group / Gender')
plt.ylabel('Diastolic BP')
plt.title('Diastolic BP by Age Group & Gender')
plt.legend(title='Gender') 

plt.show()

# Glucose and Cholesterol levels
plt.subplots_adjust(wspace=0.2, hspace=0.3)

plt.subplot(1, 2, 1)
sns.boxplot(x='encode_age', y='glucose', hue='male', data=df, palette='seismic', order=['32-40', '40-60', '>60'])
plt.xlabel('Age Group / Gender')
plt.ylabel('Glucose')
plt.title('Glucose Count by Age Group & Gender')
plt.legend(title='Gender')

# Glucose Count by Age Group & Gender : We can clearly observe that as Age increases the count of Glucose increases too.
# While Gender wise Glucose Count has almost similiar Median with Few outliers in each.

plt.subplot(1, 2, 2)
sns.boxplot(x='encode_age', y='totChol', hue='male', data=df, palette='seismic', order=['32-40', '40-60', '>60'])
plt.xlabel('Age Group / Gender')
plt.ylabel('Total Cholesterol')
plt.title('Total Cholesterol Count by Age Group')
plt.legend(title='Gender')

# Total Cholesterol by Age Group & Gender : Excluding Outliers, Observation make us Clear that for females Cholesterol level is 
# Increasing by Age considering the Quantile (25%, 50%, 75%) values into account. 
# While, for Males the Cholesterol level Quantile is Approx. Similar for each Age Group.

plt.show()

# To highlight the number of cigaretters smoked for each age group, and have an idea of the distributions of the variable, we use a violin plot
sns.catplot(data=df, x='encode_age', y='cigsPerDay', kind='violin', order=['32-40', '40-60', '>60'])
plt.xlabel('Age Group / Gender')
plt.ylabel('Cigs. / Day')
plt.xticks(ticks=[0,1,2], labels=['32-40', '40-60', '>60'])
plt.title('Cigs. per day by Age Group')
plt.show()

# In 32-40 we can observe that Median values has Lower Kernel Density followed by 75% IQR's Density. While, 25% IQR marks the Higher Kernel Density.
# In 40-60 we can observe that 25% IQR & Median has Higher Kernel Density while 75% IQR has a quite Lower Kernel Density.
# In >60 we can observe that Median and 25% IQR are Closely Intact to each other having Higher Kernel Density, while 75% IQR got Lower Kernel Density.

# Since diabetes is known to be another crucial factor in CHD, we can plot the number of diabetic individuals in the dataset, divided by age group
sns.catplot(x='encode_age', hue='diabetes', data=df, kind='count', legend=False, order=['32-40', '40-60', '>60'])
plt.xlabel('Age Group / Gender')
plt.xticks(ticks=[0,1,2], labels=['32-40', '40-60', '>60'])
plt.ylabel('No. of Patients')
plt.legend(['Negative', 'Positive'])
plt.title('Diabetes by Age Group')
plt.show()
