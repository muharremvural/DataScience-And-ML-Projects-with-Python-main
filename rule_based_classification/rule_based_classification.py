import pandas as pd

#### FIRST TASK ####

# 1
df = pd.read_csv("persona.csv")
pd.set_option("display.max_rows", None)

# 2
df["SOURCE"].nunique()
df["SOURCE"].value_counts()

# 3
df["PRICE"].nunique()

# 4
df["PRICE"].value_counts()

# 5
df["COUNTRY"].value_counts()

# 6
df.groupby("COUNTRY")["PRICE"].sum()

# 7
df["SOURCE"].value_counts()

# 8
df.groupby("COUNTRY")["PRICE"].mean()

# 9
df.groupby("SOURCE")["PRICE"].mean()

# 10
df.groupby(["COUNTRY", "SOURCE"])["PRICE"].mean()

#### SECOND TASK ####

df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"])["PRICE"].mean()

#### THIRD TASK ####

agg_df =df.groupby(["COUNTRY", "SEX", "AGE", "SOURCE"]).agg({"PRICE" : "mean"}).sort_values("PRICE", ascending=False)

#### FOURTH TASK ####

agg_df.reset_index(inplace=True)

#### FIFTH TASK ####

bins = [0, 18, 24, 30, 40, agg_df["AGE"].max()]
labels = ["0_18", "18_24", "24_30", "30_40", "40+"]

agg_df["CAT_AGE"] = pd.cut(agg_df["AGE"], bins=bins, labels=labels)

#### SIXTH TASK ####

agg_df["customers_level_based"] = agg_df[["COUNTRY", "SOURCE", "SEX", "CAT_AGE"]].agg(lambda x: "_".join(x).upper(), axis=1)
agg_df = agg_df.groupby("customers_level_based")["PRICE"].mean()
agg_df = agg_df.reset_index()
agg_df.head()

#### SEVENTH TASK ####

segments = ["D", "C", "B", "A"]
agg_df["SEGMENTS"] = pd.qcut(agg_df["PRICE"], 4, labels= segments)

#### EIGHTH TASK ####

new_user = "TUR_ANDROID_FEMALE_30_40"
new_user2 = "FRA_IOS_FEMALE_30_40"
agg_df[agg_df["customers_level_based"] == new_user]
agg_df[agg_df["customers_level_based"] == new_user2]
