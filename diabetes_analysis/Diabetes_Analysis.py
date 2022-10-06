##############################
# DIABETES FEATURE ENGINEERING
##############################

# Problem : It is desired to develop a machine learning model that can predict whether people have diabetes
# when their characteristics are specified. You are expected to perform the necessary data analysis and
# feature engineering steps before developing the model.

# The dataset is part of the large dataset held at the National Institutes of Diabetes-Digestive-Kidney Diseases in the USA.
# Data used for diabetes research on Pima Indian women aged 21 and over living in Phoenix, the 5th largest city
# of the State of Arizona in the USA. It consists of 768 observations and 8 numerical independent variables.
# The target variable is specified as "outcome"; 1 indicates positive diabetes test result, 0 indicates negative.

# Pregnancies: Number of pregnancies
# Glucose: Glucose
# BloodPressure: Diastolic Blood Pressure
# SkinThickness: Skin Thickness
# Insulin: Measured Insulin value
# BMI: Body Mass Index
# DiabetesPedigreeFunction: A function that calculates our probability of having diabetes based on our descendants.
# Age: Age(Year)
# Outcome: Whether the person has diabetes or not. Have diabetes (1) or not (0)

##############################
# TASKS
##############################

# TASK 1: EXPLORATORY DATA ANALYSIS
# Step 1: Examine the overall picture.
# Step 2: Grab numeric and categorical variables.
# Step 3: Analyze the numerical and categorical variables.
# Step 4: Perform target variable analysis. (The average of the target variable according to the categorical variables, the average of the numerical variables according to the target variable)
# Step 5: Perform outlier observation analysis.
# Step 6: Perform a missing observation analysis.
# Step 7: Perform correlation analysis.

# TASK 2: FEATURE ENGINEERING
# Step 1:  Take necessary actions for missing and outlier values. There are no missing observations in the data set,
# but Glucose, Insulin etc. Observation units containing a value of 0 in the variables may represent the missing value.
# For example; a person's glucose or insulin value will not be 0. Considering this situation,
# you can assign the zero values to the relevant values as NaN and then apply the operations to the missing values.
# Step 2: Create new variables.
# Step 3: Perform the encoding operations.
# Step 4: Standardize for numeric variables.
# Step 5: Build a model.

# Necessary libraries and functions.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv("diabetes/diabetes.csv")


##################################
# TASK 1: EXPLORATORY DATA ANALYSIS
##################################

##################################
# OVERALL PICTURE
##################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)


##################################
# GET CATEGORICAL AND NUMERICAL VARIABLES
##################################

def get_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O"]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtype != "O"]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtype == "O"]
    cat_cols = cat_cols + num_but_cat
    num_cols = [col for col in dataframe.columns if dataframe[col].dtype != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = get_col_names(df)

cat_cols
num_cols
cat_but_car


##################################
# ANALYSIS OF CATEGORICAL VARIABLES
##################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


cat_summary(df, "Outcome")


##################################
# ANALYSIS OF NUMERICAL VARIABLES
##################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, plot=True)


##################################
# TARGET VARIABLE ANALYSIS BY NUMERICAL VALUES
##################################

def target_summary_by_num(dataframe, target, num_col):
    print(dataframe.groupby(target).agg({num_col: "mean"}), end="\n\n\n")


for col in num_cols:
    target_summary_by_num(df, "Outcome", col)


##################################
# OUTLIERS ANALYSIS
##################################

def outlier_thresholds(dataframe, col, q1=0.05, q3=0.95):
    quartile1 = dataframe[col].quantile(q1)
    quartile3 = dataframe[col].quantile(q3)
    IQR = quartile3 - quartile1
    lower_th = quartile1 - 1.5 * IQR
    upper_th = quartile3 + 1.5 * IQR
    return lower_th, upper_th


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


##################################
# CORRELATION
##################################

df.corr()
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)

##################################
# TASK 2: FEATURE ENGINEERING
##################################

##################################
# FINDING AND REPLACING NULL VALUES
##################################

missing_cols = [col for col in df.columns if (df[col] == 0).any() and col not in ["Pregnancies", "Outcome"]]

for col in missing_cols:
    df[col] = np.where(df[col] == 0, np.nan, df[col])


def missing_values_table(dataframe, na_name=False):
    null_columns = [col for col in dataframe.columns if dataframe[col].isnull().any()]
    n_missing = dataframe[null_columns].isnull().sum().sort_values(ascending=False)
    null_col_ratio = (dataframe[null_columns].isnull().sum() * 100 / dataframe.shape[0]).sort_values(ascending=False)
    missing_df = pd.concat([n_missing, np.round(null_col_ratio, 2)], axis=1, keys=["n_missing", "ratio"])
    print(missing_df, end="\n")
    if na_name:
        return null_columns


null_columns = missing_values_table(df, na_name=True)

#### FILLING THE NULL VALUES WITH MEDIAN ####

df = df.apply(lambda x: x.fillna(x.median()), axis=0)


### REPLACING OUTLIERS WITH THRESHOLD ####

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(df, variable, q1, q3)
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit


for col in df.columns:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

for col in df.columns:
    print(col, check_outlier(df, col))

#####################
# CREATING NEW VARIABLES
#####################

### BY AGE
df.loc[(df["Age"] >= 21) & (df["Age"] < 50), "New_Age_Cat"] = "mature"
df.loc[df["Age"] >= 50, "New_Age_Cat"] = "senior"

### BY BMI
df.loc[(df["BMI"] <= 18.4) & (df["BMI"] >= 0), "New_BMI_Cat"] = "skinny"
df.loc[(df["BMI"] >= 18.5) & (df["BMI"] <= 24.9), "New_BMI_Cat"] = "normal"
df.loc[(df["BMI"] >= 25) & (df["BMI"] <= 29.9), "New_BMI_Cat"] = "overweight"
df.loc[(df["BMI"] >= 30) & (df["BMI"] <= 34.9), "New_BMI_Cat"] = "overweight"
df.loc[(df["BMI"] >= 34.9), "New_BMI_Cat"] = "obese"

### BY GLUCOSE
bins = [0, 80, 120, 200]
df["New_Glucose"] = pd.cut(df["Glucose"], bins=bins, labels=["Low", "Normal", "Risk"])

### BMI AND AGE

df.loc[(df["BMI"] <= 18.4) & (df["BMI"] >= 0) & (df["Age"] >= 21) & (df["Age"] < 50), "Age_BMI"] = "skinnymature"
df.loc[(df["BMI"] <= 18.4) & (df["BMI"] >= 0) & (df["Age"] >= 50), "Age_BMI"] = "skinnysenior"
df.loc[(df["BMI"] >= 18.5) & (df["BMI"] <= 24.9) & (df["Age"] >= 21) & (df["Age"] < 50), "Age_BMI"] = "normalmature"
df.loc[(df["BMI"] <= 18.5) & (df["BMI"] >= 24.9) & (df["Age"] >= 50), "Age_BMI"] = "normalsenior"
df.loc[(df["BMI"] <= 34.9) & (df["BMI"] >= 25) & (df["Age"] >= 21) & (df["Age"] < 50), "Age_BMI"] = "overweightmature"
df.loc[(df["BMI"] <= 34.9) & (df["BMI"] >= 25) & (df["Age"] >= 50), "Age_BMI"] = "overweightsenior"
df.loc[(df["BMI"] >= 34.9) & (df["Age"] >= 21) & (df["Age"] < 50), "Age_BMI"] = "obesemature"
df.loc[(df["BMI"] >= 34.9) & (df["Age"] >= 50), "Age_BMI"] = "obesesenior"


### CREATING A CATEGORY VARIABLE BASED ON INSULIN VALUES ###

def insulin_cat(dataframe, col="Insulin"):
    if 16 <= dataframe[col] <= 166:
        return "Normal"
    else:
        return "Abnormal"


df["New_Insulin_Bool"] = df.apply(insulin_cat, axis=1)

### CAPITALIZE THE COLUMNS NAMES FOR BETTER READING ###

df.columns = [col.upper() for col in df.columns]


##################################
# ENCODING
##################################

# LABEL ENCODING

def label_encoder(dataframe, binary_col):
    label_encoder = LabelEncoder()
    dataframe[binary_col] = label_encoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if (df[col].nunique() == 2) and df[col].dtype == "O"]

for col in binary_cols:
    df = label_encoder(df, col)


# ONE-HOT ENCODING
# We need to update our categoric variables first.
cat_cols, num_cols, cat_but_car = get_col_names(df)

cat_cols = [col for col in cat_cols if col not in binary_cols and col not in "OUTCOME"]

def one_hot_encoder(dataframe, cat_cols, drop_first=True):
    dataframe=pd.get_dummies(dataframe, columns=cat_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()
df.shape

##################################
# STANDARDIZING
##################################

num_cols

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()
df.shape

##################################
# MODELLING
##################################

y = df["OUTCOME"]
X = df.drop("OUTCOME", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 3)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 2)}")
print(f"F1: {round(f1_score(y_pred, y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred, y_test), 2)}")


##################################
# FEATURE IMPORTANCE
##################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value", ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')