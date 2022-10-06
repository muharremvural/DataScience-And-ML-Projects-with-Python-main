#############################
# IMPORT
#############################

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
# !pip install lightgbm
# !pip install catboost
# !pip install xgboost
# conda install -c conda-forge lightgbm
import sklearn
import seaborn as sns
import matplotlib.mlab as mlab



############ LIBRARIES ############

# BASE
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# DATA PREPROCESSING
from sklearn import preprocessing
from sklearn.neighbors import LocalOutlierFactor

# MODELING
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

import xgboost
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# MODEL TUNING
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
# WARNINGS
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

#############################
# APPENDING TRAIN AND TEST DATA SET
#############################

df_train = pd.read_csv("houseprice_prediction/train.csv")
df_test = pd.read_csv("houseprice_prediction/test.csv")
df = df_train.append(df_test).reset_index(drop=True)


#############################
# DETERMINING COLUMN TYPES
#############################


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

#############################
# ANALYZING NUMERICAL COLUMNS
#############################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

#############################
# ANALYZING CATEGORICAL COLUMNS
#############################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

#############################
# TARGET VALUE ANALYSIS
#############################

df["SalePrice"].describe([0.05, 0.1, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99]).T
sns.set(rc={'figure.figsize': (6, 6)})
df["SalePrice"].hist(bins=100)
plt.show(block=True)

# We prefer to exclude it from the dataset as 4 observations above 600000 are extreme values.
df = df.loc[~(df.SalePrice > 600000),]
df["SalePrice"].hist(bins=100)
plt.show(block=True)

### Skew
print("Skew: %f" % df["SalePrice"].skew())
### Using logarithm is a good idea in this kind of target value.
np.log1p(df["SalePrice"]).hist(bins=50)
plt.show(block=True)
print("Skew: %f" % np.log1p(df["SalePrice"]).skew())

#############################
# CORRELATION
#############################

corr = df[num_cols].corr()

sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdPu")
plt.show(block=True)

def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show(block=True)
    return drop_list

#############################
# OUTLIERS
#############################

# Defining outliers
def outlier_thresholds(dataframe, variable, low_quantile=0.10, up_quantile=0.90):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit

# Checking outliers
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

# Replacing outliers
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    if col != "SalePrice":
        replace_with_thresholds(df, col)


#############################
# MISSING VALUES
#############################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

# Some missing values represent house doesn't have that feature. It would be wrong to replace them with mean, median etc.
# Let's call this kind of columns "no_cols" and fill their null values with "No".

no_cols = ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","FireplaceQu",
           "GarageType","GarageFinish","GarageQual","GarageCond","PoolQC","Fence","MiscFeature"]

for col in no_cols:
    df[col].fillna("No", inplace=True)

# This function fills in missing values with median or mean
def quick_missing_imp(data, num_method="median", cat_length=20, target="SalePrice"):
    variables_with_na = [col for col in data.columns if
                         data[col].isnull().sum() > 0]  # Eksik değere sahip olan değişkenler listelenir

    temp_target = data[target]

    print("# BEFORE")
    print(data[variables_with_na].isnull().sum(), "\n\n")  # Uygulama öncesi değişkenlerin eksik değerlerinin sayısı

    # değişken object ve sınıf sayısı cat_lengthe eşit veya altındaysa boş değerleri mode ile doldur
    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x,
                      axis=0)

    # num_method mean ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    # num_method median ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    data[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    return data


df = quick_missing_imp(df, num_method="median", cat_length=17)

#############################
# RARE ANALYSER
#############################

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


rare_analyser(df, "SalePrice", cat_cols)


# MSZoning variable

# Indicates the Zone of the living area. Since the Residential High group is small, we can combine it with Residential Medium.
# Since the numbers of the other two groups are low, we can combine them to make it more meaningful.

df["MSZoning"].value_counts()

df.loc[(df["MSZoning"] == "RH"), "MSZoning"] = "RM"
df.loc[(df["MSZoning"] == "FV"), "MSZoning"] = "FV + C (all)"
df.loc[(df["MSZoning"] == "C (all)"), "MSZoning"] = "FV + C (all)"

df["MSZoning"].value_counts()

#LotArea variable

#It shows the ft2 of the house. There are up to 200K values, but since the majority are in lower values,
# We can group them so that it makes sense for us.

sns.set(rc={'figure.figsize': (5, 5)})
bins = 50
plt.hist(df["LotArea"],bins, alpha=0.5, density=True)
plt.show(block=True)

df["LotArea"].max()
df["LotArea"].min()

New_LotArea =  pd.Series(["Studio","Small", "Middle", "Large","Dublex","Luxury"], dtype = "category")
df["New_LotArea"] = New_LotArea
df.loc[(df["LotArea"] <= 2000), "New_LotArea"] = New_LotArea[0]
df.loc[(df["LotArea"] > 2000) & (df["LotArea"] <= 4000), "New_LotArea"] = New_LotArea[1]
df.loc[(df["LotArea"] > 4000) & (df["LotArea"] <= 6000), "New_LotArea"] = New_LotArea[2]
df.loc[(df["LotArea"] > 6000) & (df["LotArea"] <= 8000), "New_LotArea"] = New_LotArea[3]
df.loc[(df["LotArea"] > 8000) & (df["LotArea"] <= 12000), "New_LotArea"] = New_LotArea[4]
df.loc[df["LotArea"] > 12000 ,"New_LotArea"] = New_LotArea[5]

# LotShape variable

# It shows the general shape of the property. Since it has 4 groups, it is sufficient for us to have two groups as reg and IR.

# Reg Regular
# IR1 Slightly irregular
# IR2 Moderately Irregular
# IR3 Irregular

df["LotShape"].value_counts()

df.loc[df["LotShape"] == "IR1", "LotShape"] = "IR"
df.loc[df["LotShape"] == "IR2", "LotShape"] = "IR"
df.loc[df["LotShape"] == "IR3", "LotShape"] = "IR"

# ExterCond: variable

# It gives the state of the material on the exterior.

# Ex-Excellent
# Gd-Good
# TA-Average/Typical
# Fa-Fair
# Po-Poor

df["ExterCond"].value_counts()
df["ExterCond"] = np.where(df["ExterCond"].isin(["Gd", "Ex"]), "AboveTA", df["ExterCond"])
df["ExterCond"] = np.where(df["ExterCond"].isin(["Fa", "Po"]), "BelowTA", df["ExterCond"])

# GarageQual variable

# It shows the quality of the garage.

# Ex-Excellent
# Gd-Good
# TA-Typical/Average
# Fa-Fair
# Po-Poor
# NA-No Garage

df["GarageQual"].value_counts()

df["GarageQual"] = np.where(df["GarageQual"].isin(["Ex", "Gd"]), "AboveTA", df["GarageQual"])
df["GarageQual"] = np.where(df["GarageQual"].isin(["Po", "Fa", "No"]), "BelowTA", df["GarageQual"])

# BsmtFinType1 and BsmtFinType2 variable

# The quality of the finished section of the first and second basement

# GLQ Good Living Quarters
# ALQ Average Living Quarters
# BLQ Below Average Living Quarters
# Rec Average Rec Room
# LwQ Low Quality
# Unf Unfinshed
# NA No Basement

df["BsmtFinType1"].value_counts()
df["BsmtFinType2"].value_counts()

df["BsmtFinType1"] = np.where(df["BsmtFinType1"].isin(["GLQ", "ALQ"]), "AboveAVG", df["BsmtFinType1"])
df["BsmtFinType1"] = np.where(df["BsmtFinType1"].isin(["BLQ", "Rec", "LwQ"]), "BelowAVG", df["BsmtFinType1"])
df["BsmtFinType1"] = np.where(df["BsmtFinType1"].isin(["Unf"]), "No", df["BsmtFinType1"])
df["BsmtFinType2"] = np.where(df["BsmtFinType2"].isin(["Unf"]), "No", df["BsmtFinType2"])
df["BsmtFinType2"] = np.where(df["BsmtFinType2"].isin(["GLQ", "ALQ"]), "AboveAVG", df["BsmtFinType2"])
df["BsmtFinType2"] = np.where(df["BsmtFinType2"].isin(["BLQ", "Rec", "LwQ"]), "BelowAVG", df["BsmtFinType2"])

# Condition1 variable
#
# Indicates proximity to the main road or railway. We can make the adjacent ones a group, the normal a group,
# and the others a group because they are distant
#
# Artery Adjacent to arterial street
# Feedr Adjacent to feeder street
# Norm Normal
# RRNn Within 200' of North-South Railroad
# RRAn Adjacent to North-South Railroad
# PosN Near positive off-site feature--park, greenbelt, etc.
# PosA Adjacent to positive off-site feature
# RRNe Within 200' of East-West Railroad
# RRAe Adjacent to East-West Railroad

df["Condition1"].value_counts()

df["Condition1"] = np.where(df["Condition1"].isin(["Artery", "Feedr", "RRAn", "PosA", "RRAe"]), "AdjacentCondition", df["Condition1"])
df["Condition1"] = np.where(df["Condition1"].isin(["RRNe", "RRNn", "PosN"]), "WithinCondition", df["Condition1"])
df["Condition1"] = np.where(df["Condition1"].isin(["Norm"]), "NormalCondition", df["Condition1"])

# Condition2 variable
#
# Shows the second way it if there is, but it was preferred to exclude it from the data set because the diversity of the groups is very low.

df["Condition2"].value_counts()
df.drop("Condition2", axis=1, inplace=True)

# BldgType variable
#
# Gives the type of building
#
# 1Fam-Single-family Detached
# 2fmCon-Two-family Conversion; originally built as one-family dwelling
# Duplex-Duplex
# TwnhsE-Townhouse End Unit
# TwnhsI-Townhouse Inside Unit

df["BldgType"].value_counts()
df["BldgType"] = np.where(df["BldgType"].isin(["1Fam", "2fmCon"]), "FamilySize", df["BldgType"])
df["BldgType"] = np.where(df["BldgType"].isin(["Duplex", "TwnhsE", "Twnhs"]), "Large", df["BldgType"])

#############################
# CREATING NEW VARIABLES
#############################

# TotalQual variable

# Let's create a total quality indicator variable with variables indicating quality.

df["TotalQual"] = df[["OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtCond", "BsmtFinType1",
                      "BsmtFinType2", "HeatingQC", "KitchenQual", "Functional", "FireplaceQu", "GarageQual", "GarageCond", "Fence"]].sum(axis=1)

# Overall variable
# Let's create a variable from the general condition of the house and the quality of the materials used.

df["Overall"] = df[["OverallQual", "OverallCond"]].sum(axis=1)

# NEW_TotalFlrSF variable
# Total surface area of the house

df["NEW_TotalFlrSF"] = df["1stFlrSF"] + df["2ndFlrSF"]

# NEW_TotalBsmtFin variable
# Completed total basement area

df["NEW_TotalBsmtFin"] = df["BsmtFinSF1"] + df["BsmtFinSF2"]

# NEW_PorchArea variable
# Total area outside the house

df["NewPorchArea"] = df["OpenPorchSF"] + df["EnclosedPorch"] + df["ScreenPorch"] + df["3SsnPorch"] + df["WoodDeckSF"]

# NEW_TotalHouseArea variable
# total area of the house

df["NEW_TotalHouseArea"] = df["NEW_TotalFlrSF"] + df["TotalBsmtSF"]

# NEW_TotalSqFeet variable
# Total occupancy of the house ft2

df["NEW_TotalSqFeet"] = df["GrLivArea"] + df["TotalBsmtSF"]

#NEW_TotalFullBath and NEW_TotalHalfBath variables
#Number of half and full bathrooms with the total in the house

df["NEW_TotalFullBath"] = df["BsmtFullBath"] + df["FullBath"]
df["NEW_TotalHalfBath"] = df["BsmtHalfBath"] + df["HalfBath"]

# NEW_TotalBath variable
# It represents the total number of bathrooms in the house.

df["NEW_TotalBath"] = df["NEW_TotalFullBath"] + (df["NEW_TotalHalfBath"]*0.5)

# Lot Ratio variables
# How much of the land is inhabited, total house area and garage area

df["NEW_LotRatio"] = df["GrLivArea"] / df["LotArea"]

df["NEW_RatioArea"] = df["NEW_TotalHouseArea"] / df["LotArea"]

df["NEW_GarageLotRatio"] = df["GarageArea"] / df["LotArea"]

#Variables for differences between dates
#Variables such as the last year between the restoration and the year of construction,
#the difference between the year the garage was built and the year the house was built.

df["NEW_Restoration"] = df.YearRemodAdd - df.YearBuilt
df["NEW_HouseAge"] = df.YrSold - df.YearBuilt
df["NEW_RestorationAge"] = df.YrSold - df.YearRemodAdd
df["NEW_GarageAge"] = df.GarageYrBlt - df.YearBuilt
df["NEW_GarageRestorationAge"] = np.abs(df.GarageYrBlt - df.YearRemodAdd)
df["NEW_GarageSold"] = df.YrSold - df.GarageYrBlt

#
df.head()
df.shape
#

cat_cols_temp = [col for col in cat_cols if col not in "Condition2"]
rare_analyser(df, "SalePrice", cat_cols_temp)

# According to rare analyser function:
# Street, Alley, LandContour, Utilities, LandSlope, Condition2, Heating,
# CentralAir, Functional, PoolQC, MiscFeature, Neighborhood, KitchenAbvGr
# Columns don't have enough information, distrubtion is too skewed.
# It is best to remove these columns.

drop_list = ["Street", "Alley", "LandContour", "Utilities" ,"LandSlope","Heating",
             "PoolQC", "MiscFeature","Neighborhood","KitchenAbvGr", "CentralAir", "Functional"]

df.drop(drop_list, axis=1, inplace=True)

#############################
# ENCODING
#############################

# Re-defining column types with grab_col_names() func.
cat_cols, num_cols, cat_but_car = grab_col_names(df)

#############################
# LABEL ENCODER
#############################

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O"
               and len(df[col].unique()) == 2]

for col in binary_cols:
    label_encoder(df, col)

#############################
# ONE HOT ENCODER
#############################

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

#############################
# MODELLING
#############################

missing_values_table(df)

# Seperating train and test dataframes because we combined them in the first place.
train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()].drop("SalePrice", axis=1)

# It would be best to take the logarithm of the SalePrice variable,
# both because it is skewed to the right and because there is a lot of variation between units.

y = np.log1p(df[df["SalePrice"].notnull()]["SalePrice"])
X = train_df.drop(["Id", "SalePrice"], axis=1)

# Splitting train data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor())]
          # ("CatBoost", CatBoostRegressor(verbose=False))]

#RMSE
for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")


# HYPERPARAM OPTIMIZATION on XGBoost

xgboost_model = XGBRegressor(objective='reg:squarederror')

rmse = np.mean(np.sqrt(-cross_val_score(xgboost_model,
                                        X, y, cv=5, scoring="neg_mean_squared_error")))


xgboost_params = {"learning_rate": [0.1, 0.01, 0.03],
                  "max_depth": [5, 6, 8],
                  "n_estimators": [100, 200, 300],
                  "colsample_bytree": [0.5, 0.8, 1]}

xgboost_gs_best = GridSearchCV(xgboost_model,
                            xgboost_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X_train, y_train)

final_model = xgboost_model.set_params(**xgboost_gs_best.best_params_).fit(X,y)
rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))


#############################
# MODEL VALIDATION
#############################

xgboost_tuned = XGBRegressor(objective='reg:squarederror',**xgboost_gs_best.best_params_).fit(X_train, y_train)
y_pred = xgboost_tuned.predict(X_test)


# Inversing the LOG transform.
new_y= np.expm1(y_pred)
new_y_test= np.expm1(y_test)

np.sqrt(mean_squared_error(new_y_test, new_y))

# RMSE : 23574.271684387233

df['SalePrice'].mean()

submission_df = pd.DataFrame()
submission_df['Id'] = test_df["Id"].astype("Int32")

y_pred_sub = xgboost_tuned.predict(test_df.drop("Id", axis=1))

y_pred_sub = np.expm1(y_pred_sub)

submission_df['SalePrice'] = y_pred_sub
