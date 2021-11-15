import pandas as pd
housing = pd.read_csv("data.csv")


from sklearn.model_selection import StratifiedShuffleSplit
spl = StratifiedShuffleSplit(n_splits=1,test_size=.2,random_state=42)
for train_index, test_index in spl.split(housing,housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

housing = strat_train_set.copy()

housing = strat_train_set.drop('MEDV', axis=1)
housing_labels = strat_train_set['MEDV'].copy()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
# ... add as many as you want in your pipeline 
    ('std_scaler', StandardScaler()),
])

housing_num_tr = my_pipeline.fit_transform(housing)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)

from joblib import dump
dump(model, 'Dragon.joblib') 
