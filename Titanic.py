import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer

titanic_file_path = 'C:\\Users\\Dips\\Desktop\\Data Science work\\Titanic Kaggle\\train.csv'
titanic_data = pd.read_csv(titanic_file_path)

features = ['PassengerId','Pclass','Age','Fare']
X = titanic_data[features]
y = titanic_data.Survived

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(train_X))
imputed_X_val = pd.DataFrame(my_imputer.fit_transform(val_X))

# Imputation removed column names; put them back
imputed_X_train.columns = train_X.columns
imputed_X_val.columns = val_X.columns

print(titanic_data.columns)

titanic_model = RandomForestRegressor(random_state=1)
titanic_model.fit(imputed_X_train, train_y)

Titanic_predictions = titanic_model.predict(imputed_X_val)
Titanic_mae = mean_absolute_error(Titanic_predictions, val_y)

titanic_test_path = 'C:\\Users\\Dips\\Desktop\\Data Science work\\Titanic Kaggle\\test.csv'
titanic_test_data = pd.read_csv(titanic_test_path)

missing_val_count_by_column = (titanic_test_data.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

imputed_titanic_test_data = pd.DataFrame(my_imputer.fit_transform(titanic_test_data[features]))

# Imputation removed column names; put them back
imputed_titanic_test_data.columns = titanic_test_data[features].columns

titanic_test_features = titanic_test_data[features]
y_res = titanic_model.predict(imputed_titanic_test_data)

print('Name')
res = pd.DataFrame({'PassengerId': titanic_test_data.PassengerId,
                    'Survived': y_res})
res.to_csv('submission.csv',index=False)



