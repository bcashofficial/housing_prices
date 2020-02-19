import statsmodels.api as sm
from sklearn import datasets
from sklearn import linear_model
import pandas as pd

boston_raw_data = datasets.load_boston()

def load_boston_dataframe(raw_data):
	
	return pd.DataFrame(boston_raw_data.data, columns = boston_raw_data.feature_names)
	
def load_boston_target(raw_data):
	
	return pd.DataFrame(boston_raw_data.target, columns = ['MEDV'])

# ########################################################################################################################
# statsmodel does not add a constanby default. Lets see it first without a constant in our regression model.
# ########################################################################################################################

def no_constant_test_sm():
	df = load_boston_dataframe(boston_raw_data)
	target = load_boston_target(boston_raw_data)
	
	X = df['RM']
	y = target['MEDV']
	
	# Fitting a model means that you're making an algorithm learn the relationship betwen the predictors and outcome so that you can predict the future values of the outcome.
	# Note the difference in argument order. [y,X]
	model = sm.OLS(y,X).fit()
	predictions = model.predict(X)
	
	# Print out the statistics.
	print(model.summary())
	
	# Interpreting the statistics.
	
	# Dependent Variable, Model, Method: Self-Explanatory. What's the dependent variable and how are we trying to retrieve this information. OLS (Ordinary Least Squares) means
	#	we are trying to fit a regression line that would minimize the square of distance from the regression line.
	
	# Date & Time: When did we run this script.
	
	# Number of Observations (506): Self Explanatory.
	
	# DF Residuals (505) & DF Model (1) relate to degrees of freedom, which are the number of values in the final calculation of a statistic that are free to vary.
	
	# RM Coefficient (3.6534): Means that as the RM variable increases by 1, the predicted value of MDEV increases by 3.6534.
	
	# R Squared, which represents how close the data is to the fitted regression line. The coefficient of determination/coefficient of multiple determination for mltiple regression.
		# Always between 0% and 100%. The higher the R squared, the better the model fits your data, as 100% indivates the model explains all of the variability of the response data around its mean.
		# Theoretically, if a model could explain 100% of the variance, the fitted values would always equal the observed values, and therefore all of the data points would fall on the fitted regression line.
	
	# 95% confidence interval that value of RM is between 3.548 and 3.759.
	
def with_constant_test_sm():
	df = load_boston_dataframe(boston_raw_data)
	target = load_boston_target(boston_raw_data)
	
	# LSTAT was added to improve model.
		# Usually when we add variables to a regression model, R squared will be higher.
	X = df[['RM', 'LSTAT']]
	y = target['MEDV']
	
	# Adds a constant term to the linear equation it is fitting, not the values. In single predictor case, it's the difference between fitting a line y = mx to data vs fitting y = mx + b.
		# Without a constant, we are forcing our model to go through the origin, but now we have a y-intercept.
		# Tested algorithm with and without constant.
	#X = sm.add_constant(X)
	
	model = sm.OLS(y,X).fit()
	predictions = model.predict(X)
	
	print(model.summary())

# ########################################################################################################################
# SKLearn is golden standard when it comes to machine learning in Python.
# It has machine learning algorithms for regression, classification, clustering, and dimensionality reduction.
# ########################################################################################################################

def no_constant_test_sk():
	df = load_boston_dataframe(boston_raw_data)
	target = load_boston_target(boston_raw_data)
	
	# Will use all the variables this time.
	X = df 
	y = target['MEDV']
	
	lm = linear_model.LinearRegression()
	model = lm.fit(X, y)
	
	predictions = lm.predict(X)
	
	# Print the first 5 predictions of Y.
	print(predictions[0:5])
	
	# Instead of getting a pretty table like what we had from Statsmodels, we can use built in functions to return the R squared score, coefficients, and the estimated intercepts.
	print(lm.score(X,y))
	
	# Reminder that in linear regression, coefficients are the values that multiple the predictor values. The sign of each coefficient indicates the direction of the relationship between a predictor variable and response.
		# A positive sign indicates as predictor variable increases, the response increases. (Ex, if coefficient is +3, the mean response value incrases by 3 for every one unit change in the predictor.)
	print(lm.coef_)
	
	# Intercept: Expected mean value when all X is 0.
	print(lm.intercept_)

	# In practice, you would not use the entire dataset, but you will split your data into a training and test set to test model and predictions, to test model and predictions on.

# ########################################################################################################################
# R squared vs. Standard Error. 
# 	Standard provides the absolute measure of distance that data points fall from regression line.
# 	R squared provides the relative measure of percentage. R Squared will tell you a car went 80% faster without context. Standard error will say the car went 72 MPH faster.

# t-test is a type of inferential statistic used to determine if there is a significant difference between the means of two groups. The larger the t score, the more difference there is between groups. The smaller, the more similarity.
#	t score of 3 means that the groups are three times as different from each other as they are within each other.
#		3 types of T test: Comparing two groups. Comparing same group at different time intervals. One group at same time against a known value.
# p-value is a number between 0 and 1 to help determine the significane of results. A p value of .01 means there is only a 1% chance the results happened by chance.

# Every t-test has a p-value. Steps are:
#	Determine a null and alternate hypothesis.
#	Collect sample data.
#	Determine a confidence interval and degrees of freedom. This is what we call alpha. The typical value of alpha is 0.05, which means there is a 95% confidence the conclusion of this test will be valid.
#	Calculate the t-statistic.
#	Calculate the critical t value, which is the value that a test statistic must exceed for the null hypothesis to be rejected.
# 	Compare the critical t values with the calculated t statistic to know if we can confirm null hypothesis or not.

# from feature_selector import FeatureSelector -> Great tool used for feature selection.
# ########################################################################################################################
	
# no_constant_test_sm()
# with_constant_test_sm()
no_constant_test_sk()