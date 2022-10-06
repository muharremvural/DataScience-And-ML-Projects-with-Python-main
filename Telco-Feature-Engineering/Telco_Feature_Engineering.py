##############################
# TELCO CUSTOMER CHURN FEATURE ENGINEERING
##############################

# Problem: It is desired to develop a machine learning model that can predict customers who will leave the company.
# You are expected to perform the necessary data analysis and feature engineering steps before developing the model.

# Telco customer churn data includes information about a fictitious telecom company that provided home phone and Internet services
# to 7043 California customers in the third quarter. It includes which customers have left, stayed or signed up for the service.

# 21 Variables 7043 Observations

# CustomerId : Cusomter's ID
# Gender : Sex
# SeniorCitizen : Whether the customer is senior(1, 0)
# Partner : Whether the customer has a partner (Yes, No) ?
# Dependents : Whether the customer has dependents (Yes, No) (Children, Parents, Grandparents)
# tenure : Number of months the customer has stayed with the company
# PhoneService : Whether the customer has telephone service (Yes, No)
# MultipleLines : Whether the customer has more than one line (Yes, No, No phone service)
# InternetService : Customer's internet service provider (DSL, Fiber optic, No)
# OnlineSecurity : Whether the customer has online security (Yes, No, no Internet service)
# OnlineBackup : Whether the customer has an online backup (Yes, No, no Internet service)
# DeviceProtection : Whether the customer has device protection (Yes, No, no Internet service)
# TechSupport : Whether the customer has technical support (Yes, No, no Internet service)
# StreamingTV : Whether the customer has TV broadcast (Yes, No, no Internet service) Indicates whether the customer uses Internet service to stream television programs from a third-party provider
# StreamingMovies : Whether the customer is streaming movies (Yes, No, no Internet service) Indicates whether the customer is using the Internet service to stream movies from a third-party provider
# Contract : Customer's contract duration (Month to month, One year, Two years)
# PaperlessBilling : Whether the customer has a paperless invoice (Yes, No)
# PaymentMethod : Customer's payment method (Electronic check, Postal check, Bank transfer (automatic), Credit card (automatic))
# MonthlyCharges : Amount collected from the customer on a monthly basis
# TotalCharges : Total amount charged from customer
# Churn : Whether the customer is using (Yes or No) - Customers who left in the last month or quarter


# Each row represents a unique customer.
# Variables contain information about customer service, account and demographic data.
# Services customers sign up for - phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
# Customer account information – how long they have been a customer, contract, payment method, paperless invoicing, monthly fees and total fees
# Demographics about customers - gender, age range, and whether they have partners and dependents


# TASK 1: EXPLORATORY DATA ANALYSIS
           # Step 1: Examine the overall picture.
           # Step 2: Grab numerical and categorical variables.
           # Step 3: Analyse numerical and categorical variables
           # Step 4: Target value analysis.
           # Step 5: Outlier Analysis.
           # Step 6: Missing value analysis.
           # Step 7: Correlation.

# TASK 2: FEATURE ENGINEERING
           # Step 1: Take necessary actions for missing and outlier values.
           # Step 2: Create new variables based on existing variables.
           # Step 3: Encoding.
           # Step 4: Standardize numerical values.
           # Step 5: Build the model.


# Necessary libraries and functions.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
import warnings
warnings.simplefilter(action="ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv("Telco/Telco-Customer-Churn.csv")

### EXAMINE THE OVERALL PICTURE ####

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

### GRAB THE CATEGORICAL AND NUMERICAL VARIABLES ###

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtype != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtype == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtype != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)

### TotalCharges column must be numerical
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
cat_cols, num_cols, cat_but_car = grab_col_names(df)

### Converting target value to binary
df["Churn"].apply(lambda x: 1 if x=="Yes" else 0)

### CATEGORICAL VALUES ANALYSIS ###
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    print(cat_summary(df, col, plot=True))

### NUMERICAL VALUES ANALYSIS ###
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    print(cat_summary(df,col, plot=True))

## Looking at Tenure, we see that 1-month customers are too many, followed by 72-month customers.
df["tenure"].value_counts().head()
df["Churn"] = df["Churn"].apply(lambda x: 1 if x=="Yes" else 0)

## This may be due to contract types. Let's check month-by-month contracts.
df[df["Contract"] == "Month-to-month"]["tenure"].hist(bins=20)
plt.xlabel("tenure")
plt.title("Month-to-month")
plt.show(block=True)

### ANALYZE TARGET VALUE BY NUMERICAL VALUES ###

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Churn", col)

### ANALYZE TARGET VALUE BY CATEGORICAL VALUES ###

def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)

### OUTLIERS ANALYSIS ###

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

for col in num_cols:
    print(outlier_thresholds(df, col))

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(check_outlier(df, col))

# No Thresholds

### CORRELATION ####

df[num_cols].corr()

df.corr()

# Correlation Matrix
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)

### MISSING VALUES ####

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

na_cols = missing_values_table(df, na_name=True)

df[na_cols] = df[na_cols].apply(lambda x: x.fillna(x.median()), axis=0)

### CREATING NEW VARIABLES ###

#Has Streaming service or not
df.loc[(df["StreamingMovies"] == "Yes") | (df["StreamingTV"] == "Yes"), "New_anyStreamingServices"] = "Yes"
df.loc[(df["StreamingMovies"] == "No") & (df["StreamingTV"] == "No"), "New_anyStreamingServices"] = "No"

#Grouping Tenure
df.loc[(df["tenure"] <= 3) & (df["tenure"] > 0), "New_SubType"] = "new_sub"
df.loc[(df["tenure"] <= 12) & (df["tenure"] > 3), "New_SubType"] = "interested_sub"
df.loc[(df["tenure"] <= 36) & (df["tenure"] > 12), "New_SubType"] = "long_sub"
df.loc[(df["tenure"] <= 72) & (df["tenure"] > 36), "New_SubType"] = "loyal_sub"

#How many additional Services
df['New_Additional_Services'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                                     'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                     'StreamingMovies', 'StreamingTV']] == 'Yes').sum(axis=1)
#Has Auto Payment or Not
df["New_hasAutoPayment"] = df["PaymentMethod"].apply(lambda x: 1 if x in
                                                                    ["Credit card (automatic)", "Bank transfer (automatic)"] else 0)
#Average Charges per month
df["New_AVG_Charges"] = df["TotalCharges"] / (df["tenure"] + 0.01)

# Ratio of avg charges compared to monthly charges
df["New_AVG_Increase"] = df["MonthlyCharges"] / (df["MonthlyCharges"] + 1)

# Charges per each additional services.
df["New_AVG_AddServices"] = df["MonthlyCharges"] / df["New_Additional_Services"] + 0.1

df.shape
df.head()

#### ENCODING

### Categorizing the variables because of the new ones we created.
cat_cols, num_cols, cat_but_car = grab_col_names(df)

### Label Encoding

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in cat_cols if df[col].nunique() == 2 and df[col].dtype == "O"]

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_cols:
    df = label_encoder(df, col)

### One-Hot Encoding ####

### Updating cat_cols
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn", "New_Additional_Services"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

#### BUILDING THE MODEL ####

y = df["Churn"]
X = df.drop(["Churn", "customerID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),2)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")


#### PLOT OF IMPORTANCE ####


def plot_feature_importance(importance,names,model_type):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(25, 10))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Add chart labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.show(block=True)


plot_feature_importance(catboost_model.get_feature_importance(), X.columns, 'CATBOOST')







