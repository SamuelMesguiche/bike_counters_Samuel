# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-11-29T12:54:50.749580Z","iopub.execute_input":"2023-11-29T12:54:50.750071Z","iopub.status.idle":"2023-11-29T12:55:01.376531Z","shell.execute_reply.started":"2023-11-29T12:54:50.750038Z","shell.execute_reply":"2023-11-29T12:55:01.374883Z"}}
#!/usr/bin/env python
# coding: utf-8

# # RAMP on predicting cyclist traffic in Paris
# 
# Authors: *Roman Yurchak (Symerio)*; also partially inspired by the air_passengers starting kit.
# 
# 
# ## Introduction
# 
# The dataset was collected with cyclist counters installed by Paris city council in multiple locations. It contains hourly information about cyclist traffic, as well as the following features,
#  - counter name
#  - counter site name
#  - date
#  - counter installation date
#  - latitude and longitude
#  
# Available features are quite scarce. However, **we can also use any external data that can help us to predict the target variable.** 

# In[1]:


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn


# In[2]:


data_train = pd.read_parquet(Path("/kaggle/input/mdsb-2023/train.parquet"))


# In[3]:


data_train


# We can check general information about different columns,

# In[4]:


data_train.info()


# and in particular the number of unique entries in each column,

# In[5]:


data_train.nunique(axis=0)


# We have a 30 counting sites where sometimes multiple counters are installed per location.  Let's look at the most frequented stations,

# In[6]:


data_train.groupby(["site_name", "counter_name"], observed=True)["log_bike_count"].sum().sort_values(ascending = False).to_frame()


# # Visualizing the data
# 
# 
# Let's visualize the data, starting from the spatial distribution of counters on the map

# In[7]:


import folium

m = folium.Map(location=data_train[["latitude", "longitude"]].mean(axis=0), zoom_start=13)

for _, row in (
    data_train[["counter_name", "latitude", "longitude"]]
    .drop_duplicates("counter_name")
    .iterrows()
):
    folium.Marker(
        row[["latitude", "longitude"]].values.tolist(), popup=row["counter_name"]
    ).add_to(m)

m


# Note that in this RAMP problem we consider only the 30 most frequented counting sites, to limit data size.
# 
# 
# Next we will look into the temporal distribution of the most frequented bike counter. If we plot it directly we will not see much because there are half a million data points,

# In[8]:


#mask = data_train["counter_name"] == "Totem 73 boulevard de Sébastopol S-N"

#data_train[mask].plot(x="date", y="bike_count")


# Instead we aggregate the data, for instance, by week to have a clearer overall picture,

# In[9]:


mask = data_train["counter_name"] == "Totem 73 boulevard de Sébastopol S-N"

data_train[mask].groupby(pd.Grouper(freq="1w", key="date"))[["bike_count"]].sum().plot()


# In[10]:


# We want to aggregate more counters on the same plot, we think that 7 is a good number to observe a trend.
# Find the top 10 counters by total bike count
top_counters = data_train.groupby("counter_name", observed = True)["bike_count"].sum().nlargest(7).index

# Create a mask for these top counters
mask = data_train["counter_name"].isin(top_counters)

# Group by week and plot
data_train[mask].groupby(["counter_name", pd.Grouper(freq="1w", key="date")], observed = True)["bike_count"].sum().unstack(0).plot(figsize = (12,6), legend = 'upper right')



# We can see that there is a real trend based on the period, we will see why? It is probably a period of lockdown or holidays.

# While at the same time, we can zoom on a week in particular for a more short-term visualization,

# In[11]:


fig, ax = plt.subplots(figsize=(10, 4))

mask = (
    (data_train["counter_name"] == "27 quai de la Tournelle SE-NO")
    & (data_train["date"] > pd.to_datetime("2021/06/01"))
    & (data_train["date"] < pd.to_datetime("2021/06/08"))
)

data_train[mask].plot(x="date", y="bike_count", ax=ax)


# The hourly pattern has a clear variation between work days and weekends (7 and 8 March 2021).
# 
# If we look at the distribution of the target variable it skewed and non normal, 

# In[12]:


top_counters = data_train.groupby("counter_name", observed = True)["bike_count"].sum().nlargest(4).index

# Time frame
start_date = pd.to_datetime("2021/03/01")
end_date = pd.to_datetime("2021/03/08")

# Initialize plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot each of the top counters
for counter in top_counters:
    mask = (
        (data_train["counter_name"] == counter) &
        (data_train["date"] > start_date) &
        (data_train["date"] < end_date)
    )
    data_train[mask].plot(x="date", y="bike_count", ax=ax, label=counter)

# Customize the plot
plt.legend(fontsize='small')
plt.show()


# We can see that there are some counters where the peaks of the day are often in the moring while for some there often at night

# In[13]:


import seaborn as sns


ax = sns.histplot(data_train, x="bike_count", kde=True, bins=50)


# Least square loss would not be appropriate to model it since it is designed for normal error distributions. One way to precede would be to transform the variable with a logarithmic transformation,
# ```py
# data['log_bike_count'] = np.log(1 + data['bike_count'])
# ```

# In[14]:


ax = sns.histplot(data_train, x="log_bike_count", kde=True, bins=50)


# which has a more pronounced central mode, but is still non symmetric. In the following, **we use `log_bike_count` as the target variable** as otherwise `bike_count` ranges over 3 orders of magnitude and least square loss would be dominated by the few large values. 

# ## Feature extraction
# 
# To account for the temporal aspects of the data, we cannot input the `date` field directly into the model. Instead we extract the features on different time-scales from the `date` field, 

# In[15]:


def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])


# In[16]:


data_train["date"].head()


# In[17]:


_encode_dates(data_train[["date"]])


# To use this function with scikit-learn estimators we wrap it with [FunctionTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html),

# In[18]:


from sklearn.preprocessing import FunctionTransformer

date_encoder = FunctionTransformer(_encode_dates, validate=False)
date_encoder.fit_transform(data_train[["date"]]).head()


# Since it is unlikely that, for instance, that `hour` is linearly correlated with the target variable, we would need to additionally encode categorical features for linear models. This is classically done with [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html), though other encoding strategies exist.

# In[19]:


from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(sparse_output=False)

enc.fit_transform(_encode_dates(data_train[["date"]])[["hour"]].head())


# ## Linear model

# Let's now construct our first linear model with [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html). We use a few helper functions defined in `problem.py` of the starting kit to load the public train and test data:

# In[20]:


data_test = pd.read_parquet(Path("/kaggle/input/mdsb-2023/final_test.parquet"))


# In[21]:


data_test


# In[22]:


from sklearn.model_selection import TimeSeriesSplit

_target_column_name = "log_bike_count"

def get_cv(X, y, random_state=0):
    cv = TimeSeriesSplit(n_splits=8)
    rng = np.random.RandomState(random_state)

    for train_idx, test_idx in cv.split(X):
        yield train_idx, rng.choice(test_idx, size=len(test_idx) // 3, replace=False)

def prepare_data(data):
    # Sort by date first
    data = data.sort_values(["date", "counter_name"])
    
    # Check if target column exists in the dataset
    if _target_column_name in data.columns:
        y_array = data[_target_column_name].values
        X_df = data.drop([_target_column_name], axis=1)
    else:
        y_array = None
        X_df = data

    # Drop 'bike_count' if it exists
    if 'bike_count' in X_df.columns:
        X_df = X_df.drop(['bike_count'], axis=1)

    return X_df, y_array

# Preparing the datasets
X_train, y_train = prepare_data(data_train)
X_test, y_test = prepare_data(data_test) 


# In[23]:


X_train


# and

# In[24]:


y_train


# Where `y` contains the `log_bike_count` variable. 
# 
# The test set is in the future as compared to the train set,

# In[25]:


print(
    f'Train: n_samples={X_train.shape[0]},  {X_train["date"].min()} to {X_train["date"].max()}'
)
print(
    f'Test: n_samples={X_test.shape[0]},  {X_test["date"].min()} to {X_test["date"].max()}'
)


# In[26]:


_encode_dates(X_train[["date"]]).columns.tolist()


# In[27]:


from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

date_encoder = FunctionTransformer(_encode_dates)
date_cols = _encode_dates(X_train[["date"]]).columns.tolist()

categorical_encoder = OneHotEncoder(handle_unknown="ignore")
categorical_cols = ["counter_name", "site_name"]

preprocessor = ColumnTransformer(
    [
        ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
        ("cat", categorical_encoder, categorical_cols),
    ]
)

regressor = Ridge()

pipe = make_pipeline(date_encoder, preprocessor, regressor)
pipe.fit(X_train, y_train)


# We then evaluate this model with the RMSE metric,

# In[28]:


from sklearn.metrics import mean_squared_error

print(
    f"Train set, RMSE={mean_squared_error(y_train, pipe.predict(X_train), squared=False):.2f}"
)
#print(
#    f"Test set, RMSE={mean_squared_error(y_test, pipe.predict(X_test), squared=False):.2f}"
#)


# The model doesn't have enough capacity to generalize on the train set, since we have lots of data with relatively few parameters. However it happened to work somewhat better on the test set. We can compare these results with the baseline predicting the mean value,

# In[29]:


print("Baseline mean prediction.")
print(
    f"Train set, RMSE={mean_squared_error(y_train, np.full(y_train.shape, y_train.mean()), squared=False):.2f}"
)
#print(
#    f"Test set, RMSE={mean_squared_error(y_test, np.full(y_test.shape, y_test.mean()), squared=False):.2f}"
#)


# which illustrates that we are performing better than the baseline.
# 
# Let's visualize the predictions for one of the stations,

# In[30]:


mask = (
    (X_test["counter_name"] == "Totem 73 boulevard de Sébastopol S-N")
    & (X_test["date"] > pd.to_datetime("2021/09/01"))
    & (X_test["date"] < pd.to_datetime("2021/09/08"))
)

#df_viz = X_test.loc[mask].copy()
#df_viz["bike_count"] = np.exp(y_test[mask.values]) - 1
#df_viz["bike_count (predicted)"] = np.exp(pipe.predict(X_test[mask])) - 1


# In[31]:


#fig, ax = plt.subplots(figsize=(12, 4))

#df_viz.plot(x="date", y="bike_count", ax=ax)
#df_viz.plot(x="date", y="bike_count (predicted)", ax=ax, ls="--")
#ax.set_title("Predictions with Ridge")
#ax.set_ylabel("bike_count")


# So we start to see the daily trend, and some of the week day differences are accounted for, however we still miss the details and the spikes in the evening are under-estimated.
# 
# A useful way to visualize the error is to plot `y_pred` as a function of `y_true`,

# In[32]:


#fig, ax = plt.subplots()

#df_viz = pd.DataFrame({"y_true": y_test, "y_pred": pipe.predict(X_test)}).sample(
#    10000, random_state=0
#)

#df_viz.plot.scatter(x="y_true", y="y_pred", s=8, alpha=0.1, ax=ax)


# It is recommended to use cross-validation for hyper-parameter tuning with [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) or more reliable model evaluation with [cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score). In this case, because we want the test data to always be in the future as compared to the train data, we can use [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html),
# 
# <img src="https://i.stack.imgur.com/Q37Bn.png" />
# 
# The disadvantage, is that we can either have the training set size be different for each fold which is not ideal for hyper-parameter tuning (current figure), or have constant sized small training set which is also not ideal given the data periodicity. This explains that generally we will have worse cross-validation scores than test scores, 

# In[33]:


from sklearn.model_selection import TimeSeriesSplit, cross_val_score

cv = TimeSeriesSplit(n_splits=6)

# When using a scorer in scikit-learn it always needs to be better when smaller, hence the minus sign.
scores = cross_val_score(
    pipe, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error"
)
print("RMSE: ", scores)
print(f"RMSE (all folds): {-scores.mean():.3} ± {(-scores).std():.3}")


# In[34]:


y_pred = pipe.predict(X_test)
results = pd.DataFrame(
    dict(
        Id=np.arange(y_pred.shape[0]),
        log_bike_count=y_pred,
    )
)
results.to_csv("submission.csv", index=False)


# In[ ]: