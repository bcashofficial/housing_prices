import numpy as np
import pandas as pd
import sqlalchemy
import os
import data_methods
import warnings
from scipy import stats

from sklearn.linear_model import Ridge, RidgeCV, LassoCV, ElasticNet
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
#from lightgbm import LGBMRegressor

# Prerequisite settings.
warnings.filterwarnings('ignore')

def dummies_pivot(df):
	dummies_pivot_df = pd.DataFrame()
	object_columns = df.select_dtypes(include='object').columns
	
	for column in object_columns:
		dummies = pd.get_dummies(df[column], drop_first = False)
		dummies = dummies.add_prefix('{}_'.format(column))
		dummies_pivot_df = pd.concat([dummies_pivot_df, dummies], axis = 1)
	
	return dummies_pivot_df
	
def clean_dataset(df):
	df.dropna(inplace=True)
	indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
	
	return df[indices_to_keep].astype(np.float64)

def run_model():
	
	###########################################################################################################################################################################
	# SECTION: What server are we running on?
	###########################################################################################################################################################################

	# Retrieve connection string based on server.
	mysql_connection_string = data_methods.connect_to_environment()
	#print(mysql_connection_string)

	# Connect to the database.
	try:
		database_engine = sqlalchemy.create_engine(mysql_connection_string)
		sql_engine = data_methods.connect_to_database(mysql_connection_string)
		database_connection = sql_engine.raw_connection()
		database_cursor = database_connection.cursor()
		print('Connected to database.')
	except Exception as error:
		print(repr(error))
		
	###########################################################################################################################################################################
	# SECTION: Core Logic
	###########################################################################################################################################################################
	
	# Pandas Max Rows
	pandas_max_rows_query = "select variable from core.variables where variable_group = 'HOUSING_PRICES_VARIABLES' and variable_keycode = 'pandas_max_rows' and `status` = 'ACTIVE' and file_location = 'elastic_net/housing_prices.py'"
	pandas_max_rows = data_methods.retrieve_result(database_cursor, pandas_max_rows_query)
	pd.set_option('display.max_rows', int(pandas_max_rows))
		
	# Get the imported data folder name.
	imported_data_folder_name_query = "select variable from core.variables where variable_group = 'HOUSING_PRICES_VARIABLES' and variable_keycode = 'IMPORTED_DATA_FOLDER_NAME' and `status` = 'ACTIVE' and file_location = 'elastic_net/housing_prices.py'"
	imported_data_folder_name = data_methods.retrieve_result(database_cursor, imported_data_folder_name_query)

	# Get the ignore pattern strings, so we only ingest proper files from imported data/
	file_ignore_pattern_string_query = "select group_concat(variable) from core.variables where variable_group = 'HOUSING_PRICES_EXCLUDE_PATTERN' and `status` = 'ACTIVE' and file_location = 'elastic_net/housing_prices.py';"
	file_ignore_pattern_string = data_methods.retrieve_result(database_cursor, file_ignore_pattern_string_query)

	# Split the exclude types and strip white spaces.
	file_ignore_patterns = file_ignore_pattern_string.split(',')
	file_ignore_patterns = [x.strip(' ') for x in file_ignore_patterns]
	
	# Where the final files will be stored.
	appended_files = []

	# Before excluding - load all of the files.
	for root_name, _, file_names in os.walk(imported_data_folder_name):
		for filename in file_names:
			appended_file = os.path.join(root_name, filename)
			appended_files.append(appended_file)

	# Only add the files that do not match a file ignore pattern.
	for file_ignore in file_ignore_patterns:
		appended_files = [x for x in appended_files if file_ignore not in x]

	# print(appended_files)
	
	# Load training set & test set.
	train_word_to_search_query = "select variable from core.variables where variable_group = 'HOUSING_PRICES_VARIABLES' and variable_keycode = 'train_variable' and `status` = 'ACTIVE' and file_location = 'elastic_net/housing_prices.py'"
	train_word = data_methods.retrieve_result(database_cursor, train_word_to_search_query)
	
	for csv in appended_files:
		if train_word in csv:
			train = pd.read_csv(csv)
			
			#print(train.head())
	
	test_word_to_search_query = "select variable from core.variables where variable_group = 'HOUSING_PRICES_VARIABLES' and variable_keycode = 'test_variable' and `status` = 'ACTIVE' and file_location = 'elastic_net/housing_prices.py'"
	test_word = data_methods.retrieve_result(database_cursor, test_word_to_search_query)
	
	for csv in appended_files:
		if test_word in csv:
			test = pd.read_csv(csv)
			
			#print(test.head())
			
	# Check shape of data before concat of two dataframes.
	#print ('Train data has {} rows & {} features: '.format(train.shape[0], train.shape[1]))
	#print ('Test data has {} rows & {} features: '.format(test.shape[0], test.shape[1]))
	
	main_df = pd.concat([train, test], axis=0)
	#print ('The data has {} rows & {} features: '.format(main_df.shape[0], main_df.shape[1]))
	
	# See Data Columns & Info About Data
	#print(main_df.columns)
	
	# See all of the different data types.
	#print(main_df.dtypes)
	
	# Separate numerical and categorical features.
	numerical_features = main_df.select_dtypes(include=['int64', 'float64'])
	categorical_features = main_df.select_dtypes(include='object')
	
	# Describe numerical and categorical features. (Nothing looks too off).
	#print(numerical_features.describe())
	#print(categorical_features.describe())
	
	# See all columns with the number of missing values. We have 34 columns with NaN values. Let's know what each of these columns mean so we know how to handle their missing values.
	# We will either impute the median, impute the mean, impute the mode, create a new class for OTHER for certain categorical features, set to a specific value, or drop the ID and certain unnecessary features.
	#print(main_df.isnull().sum().sort_values(ascending=False)[:40])
	
	# Fields that can be dropped.
	main_df = main_df.drop(columns=['Id'], axis=1)
	
	# LotFrontage - Impute the median grouping by Neighborhood.
	main_df['LotFrontage'] = main_df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
	
	# Set missing values to NA. Will also need to transform this field to a numeric field.
	# GarageCond, GarageQual, GarageFinish, GarageType, BsmtCond, BsmtExposure, BsmtQual, BsmtFinType1, BsmtFinType2, POOLQC, Fence, FireplaceQu
	impute_na_query = "select group_concat(variable) from core.variables where variable_group = 'HOUSING_PRICES_IMPUTE_NA' and `status` = 'ACTIVE' and file_location = 'elastic_net/housing_prices.py';"
	impute_na_string = data_methods.retrieve_result(database_cursor, impute_na_query)

	# Split the columns and strip white spaces.
	impute_na_columns = impute_na_string.split(',')
	impute_na_columns = [x.strip(' ') for x in impute_na_columns]
	
	for name in impute_na_columns:
		main_df[name].fillna('NA', inplace=True)
	
	# Set missing values to None. Will also need to transform this field to a numeric field.
	# MasVnrType, MiscFeature, Alley
	impute_none_query = "select group_concat(variable) from core.variables where variable_group = 'HOUSING_PRICES_IMPUTE_NONE' and `status` = 'ACTIVE' and file_location = 'elastic_net/housing_prices.py';"
	impute_none_string = data_methods.retrieve_result(database_cursor, impute_none_query)

	# Split the columns and strip white spaces.
	impute_none_columns = impute_none_string.split(',')
	impute_none_columns = [x.strip(' ') for x in impute_none_columns]
	
	for name in impute_none_columns:
		main_df[name].fillna('None', inplace=True)
	
	# MSZoning - Set missing values to Mode grouping by MSSubClass.
	main_df['MSZoning'] = main_df.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
	
	# Set missing values to Other. Will also need to transform this field to a numeric field.
	# Utilities, Functional, Exterior1st, Exterior2nd, TotalBsmtSF, Electrical, KitchenQual
	impute_other_query = "select group_concat(variable) from core.variables where variable_group = 'HOUSING_PRICES_IMPUTE_OTHER' and `status` = 'ACTIVE' and file_location = 'elastic_net/housing_prices.py';"
	impute_other_string = data_methods.retrieve_result(database_cursor, impute_other_query)

	# Split the columns and strip white spaces.
	impute_other_columns = impute_other_string.split(',')
	impute_other_columns = [x.strip(' ') for x in impute_other_columns]
	
	for name in impute_other_columns:
		main_df[name].fillna('Other', inplace=True)
	
	# SaleType - Set to Oth (Consistent with the verbage used for this column). Will also need to transform this field to a numeric field.
	main_df['SaleType'].fillna('0th', inplace=True)
	
	# Set Values to 0.	
	# Get all of the columns that we are transforming to zero.
	impute_zero_query = "select group_concat(variable) from core.variables where variable_group = 'HOUSING_PRICES_IMPUTE_ZERO' and `status` = 'ACTIVE' and file_location = 'elastic_net/housing_prices.py';"
	impute_zero_string = data_methods.retrieve_result(database_cursor, impute_zero_query)

	# Split the columns and strip white spaces.
	impute_zero_columns = impute_zero_string.split(',')
	impute_zero_columns = [x.strip(' ') for x in impute_zero_columns]
	
	for name in impute_zero_columns:
		main_df[name].fillna(0, inplace=True)
	
	# Check nulls after. Good - no null values.	
	# print(main_df.isnull().sum().sort_values(ascending=False)[:40])
	
	# View DF in Database
	#data_methods.write_df_to_sql(main_df, 'ReviewHousingPrices', sql_engine, 'review', 'replace')
	
	# Now its time to transform categorical features into numerical features.
	# Check object columns.
	object_features = main_df.select_dtypes(include='object').columns
	#print(object_features)
	
	# Categorical Features. (P for pivoted). (M for manual).
	# ['MSZoning' (P), 'Street' (P), 'Alley' (P), 'LotShape' (P), 'LandContour' (P), 'Utilities' (P), 'LotConfig' (P), 'LandSlope' (P), 'Neighborhood' (P), 'Condition1' (P), 'Condition2' (P), 
	# 'BldgType' (P), 'HouseStyle' (P), 'RoofStyle' (P), 'RoofMatl' (P), 'Exterior1st' (P), 'Exterior2nd' (P), 'MasVnrType' (P), 'ExterQual' (P), 'ExterCond' (P), 'Foundation' (P),
	# 'BsmtQual' (P), 'BsmtCond' (P), 'BsmtExposure' (P), 'BsmtFinType1' (P), 'BsmtFinType2' (P), 'Heating' (P), 'HeatingQC' (P), 'CentralAir' (P), 'Electrical' (P),
	# 'KitchenQual' (P), 'Functional' (P), 'FireplaceQu' (P), 'GarageType' (P), 'GarageFinish' (P), 'GarageQual' (P), 'GarageCond' (P), 'PavedDrive' (P), 
	# 'PoolQC' (P), 'Fence' (P), 'MiscFeature' (P), 'SaleType' (P), 'SaleCondition' (P)]
	
	dummies_data = dummies_pivot(main_df)
	#print(dummies_data.shape)
	
	# We can now drop the columns we pivoted and remake our final data frame with the pivoted columns in addition with the already numeric columns.
	main_df = main_df.drop(object_features, axis=1)
	final_data = pd.concat([main_df, dummies_data], axis=1)
	#print(final_data.shape)
			
	# Detect outliers in data set. * IF a Z-score is 0, it indicates that the data point's score is identical to the mean score.
	# For each column, we compute the Z-score of each value in the column, relative to the column mean and standard deviation.
	# Then we take the absolute of Z-score because direction does not matter. Only if it is below the threshold (3 in number of standard deviations away from the mean).
	# Axis = 1 ensures that for each row, all columns satisfy the constraint.
	# Finally, result of this condition is used to index the dataframe.
	final_data[(np.abs(stats.zscore(final_data)) < 3).all(axis=1)]
	
	# Add Sale Price to end of dataframe.
	final_data_columns = list(final_data.columns.values)
	final_data_columns.pop(final_data_columns.index('SalePrice'))
	final_data = final_data[final_data_columns + ['SalePrice']]
	
	# Remove nulls from SalePrice
	final_data['SalePrice'].fillna((final_data['SalePrice'].mean()), inplace=True)
	
	# View DF in Database before we start machine learning logic.
	print('Writing final data set to the database for SQL review.')
	data_methods.write_df_to_sql(final_data, 'ReviewHousingPrices', sql_engine, 'review', 'replace')
	print('Finished writing final data set to the database for SQL review.')
	
	# Create training and test set. Also set null values to median.
	#X = final_data.loc[:, final_data.columns == '3SsnPorch']
	X = final_data.loc[:, final_data.columns != 'SalePrice']
	clean_dataset(X)
	
	y = final_data['SalePrice']
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=False)
	
	# Check for missing data.
	#print(np.isnan(X.any())) #and gets False
	#print(np.isfinite(X.all())) #and gets True
	#print(np.all(np.isfinite(X)))
	
	# Ridge ML
	print('Running Ridge ML model.')
	ridge = Ridge(alpha=10, solver='auto')
	print('Fitting Ridge ML model.')
	ridge.fit(X_train, y_train)
	print('Predicting Ridge ML model.')
	ridge_predictions = ridge.predict(X_test)
	print('Finished running Ridge ML model.')
	
	# Ridge CV ML
	print('Running Ridge CV ML model.')
	ridge_cv = RidgeCV(alphas=(0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10))
	print('Fitting Ridge CV ML model.')
	ridge_cv.fit(X_train, y_train)
	print('Predicting Ridge CV ML model.')
	ridge_cv_predictions = ridge_cv.predict(X_test)
	print('Finished running Ridge CV ML model.')
	
	# Lasso CV ML
	print('Running Lasso CV ML model.')
	lasso_cv = LassoCV(alphas=(0.0001, 0.0005, 0.001, 0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10))
	print('Fitting Lasso CV ML model.')
	lasso_cv.fit(X_train, y_train)
	print('Predicting Lasso CV ML model.')
	lasso_cv_predictions = lasso_cv.predict(X_test)
	print('Finished running Lasso CV ML model.')
	
	# Elastic Net ML
	print('Running Elastic Net model.')
	elastic_net = ElasticNet(random_state=1, alpha=0.00065, max_iter=3000)
	print('Fitting Elastic Net ML model.')
	elastic_net.fit(X_train, y_train)
	print('Predicting Elastic Net ML model.')
	elastic_net_predictions = elastic_net.predict(X_test)
	print('Finished running Elastic Net ML model.')
	
	# XGBRegressor
	print('Running XGBRegressor model.')
	xgb_regressor = xgb.XGBRegressor(learning_rate=0.01, n_estimators=3460, max_depth=3, min_child_weight=0, gamma=0, subsample=0.7, colsample_bytree=0.7,
		objective='reg:squarederror', nthread=-1, scale_pos_weight=1, seed=27, reg_alpha=0.00006)
	print('Fitting XGBRegressor model.')
	xgb_regressor.fit(X_train, y_train)
	print('Predicting XGBRegressor model.')
	xgb_regressor_predictions = xgb_regressor.predict(X_test)
	print('Finished running XBGRegressor model.')
	
	# Gradient Boosting Regressor - Slower version of XGBRegressor. (Gradient Boosting - Learning from previous predictions)
	print('Running Gradient Boosting Regressor model.')
	gradient_boosting_regressor = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=42)
	print('Fitting Gradient Boosting Regressor model.')
	gradient_boosting_regressor.fit(X_train, y_train)
	print('Predicting Gradient Boosting Regressor model.')
	gradient_boosting_regressor_predictions = gradient_boosting_regressor.predict(X_test)
	print('Finished running Gradient Boosting Regressor model.')
	
	# LGBMRegressor
	#print('Running LGBMRegressor model.')
	#lgbm_regressor = GradientBoostingRegressor(objective='regression', num_leaves=4, learning_rate=0.01, n_estimators=5000, max_bin=200, bagging_fraction=0.75, bagging_freq=5, bagging_seed=7,
		#feature_fraction=0.2, feature_fraction_seed=7, verbose=1)
	#print('Fitting LGBMRegressor model.')
	#lgbm_regressor.fit(X_train, y_train)
	#print('Predicting LGBMRegressor model.')
	#lgbm_regressor_predictions = lgbm_regressor.predict(X_test)
	#print('Finished running LGBMRegressor model.')
	
	# Make final predictions. Using stacking (stacked generalizations)
	print('Making final predictions.')
	final_predictions = (0.6 * gradient_boosting_regressor_predictions) + (0.1 * xgb_regressor_predictions) + (0.3 * elastic_net_predictions) # + (0.3 * lgbm_regressor_predictions)
	print('Made final predictions.')
	
	submission = {
		'ID': test.Id.values, 
		'SalePrice': final_predictions
	}
	
	solution = pd.DataFrame.from_dict(submission, orient='index').transpose()
	print(solution.head())
	
	print('Writing solution data set to the database.')
	data_methods.write_df_to_sql(solution, 'ReviewPredictions', sql_engine, 'review', 'replace')
	print('Finished writing solution data set to the database.')
	
	###########################################################################################################################################################################
	# SECTION: End Core Logic
	###########################################################################################################################################################################

	# Close Connection	
	try:
		data_methods.close_database(database_connection)
		print('Disconnected from database.')
	except Exception as error:
		print(repr(error))

run_model()
