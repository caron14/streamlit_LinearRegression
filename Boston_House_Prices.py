import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.datasets import load_boston
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# title
st.title('Linear regression on Boston house prices')

# Read the dataset
dataset = load_boston()
df = pd.DataFrame(dataset.data)
# Assign the columns into df
df.columns = dataset.feature_names
# Assign the target variable(house prices)
df["PRICES"] = dataset.target

# Show the table data
if st.checkbox('Show the dataset as table data'):
	st.dataframe(df)

# Explanatory variable
FeaturesName = [\
              #-- "Crime occurrence rate per unit population by town"
              "CRIM",\
              #-- "Percentage of 25000-squared-feet-area house"
              'ZN',\
              #-- "Percentage of non-retail land area by town"
              'INDUS',\
              #-- "Index for Charlse river: 0 is near, 1 is far"
              'CHAS',\
              #-- "Nitrogen compound concentration"
              'NOX',\
              #-- "Average number of rooms per residence"
              'RM',\
              #-- "Percentage of buildings built before 1940"
              'AGE',\
              #-- 'Weighted distance from five employment centers'
              "DIS",\
              ##-- "Index for easy access to highway"
              'RAD',\
              ##-- "Tax rate per $100,000"
              'TAX',\
              ##-- "Percentage of students and teachers in each town"
              'PTRATIO',\
              ##-- "1000(Bk - 0.63)^2, where Bk is the percentage of Black people"
              'B',\
              ##-- "Percentage of low-class population"
              'LSTAT',\
              ]


# Check an exmple,  "Target" vs each variable
if st.checkbox('Show the relation between "Target" vs each variable'):
	checked_variable = st.selectbox(
		'Select one variable:',
		FeaturesName
		)
	# Plot
	fig, ax = plt.subplots(figsize=(5, 3))
	ax.scatter(x=df[checked_variable], y=df["PRICES"])
	plt.xlabel(checked_variable)
	plt.ylabel("PRICES")
	st.pyplot(fig)

"""
## Preprocessing
"""
# Select the variables NOT to be used
Features_chosen = []
Features_NonUsed = st.multiselect(
	'Select the variables NOT to be used', 
	FeaturesName)

df = df.drop(columns=Features_NonUsed)

# Perform the logarithmic transformation
left_column, right_column = st.beta_columns(2)
bool_log = left_column.radio(
			'Perform the logarithmic transformation?', 
			('No','Yes')
			)

df_log, Log_Features = df.copy(), []
if bool_log == 'Yes':
	Log_Features = right_column.multiselect(
					'Select the variables you perform the logarithmic transformation', 
					df.columns
					)
	# Perform logarithmic transformation
	df_log[Log_Features] = np.log(df_log[Log_Features])


# Perform the standardization
left_column, right_column = st.beta_columns(2)
bool_std = left_column.radio(
			'Perform the standardization?', 
			('No','Yes')
			)

df_std = df_log.copy()
if bool_std == 'Yes':
	Std_Features_chosen = []
	Std_Features_NonUsed = right_column.multiselect(
					'Select the variables NOT to be standardized (categorical variables)', 
					df_log.drop(columns=["PRICES"]).columns
					)
	for name in df_log.drop(columns=["PRICES"]).columns:
		if name in Std_Features_NonUsed:
			continue
		else:
			Std_Features_chosen.append(name)
	# Perform standardization
	sscaler = preprocessing.StandardScaler()
	sscaler.fit(df_std[Std_Features_chosen])
	df_std[Std_Features_chosen] = sscaler.transform(df_std[Std_Features_chosen])

"""
### Split the dataset
"""
left_column, right_column = st.beta_columns(2)
test_size = left_column.number_input(
				'Validation-dataset size (rate: 0.0-1.0):',
				min_value=0.0,
				max_value=1.0,
				value=0.2,
				step=0.1,
				 )
random_seed = right_column.number_input('Set random seed (0-):',
							  value=0, step=1,
							  min_value=0)

X_train, X_val, Y_train, Y_val = train_test_split(
	df_std.drop(columns=["PRICES"]), 
	df_std['PRICES'], 
	test_size=test_size, 
	random_state=random_seed
	)

# Create a regression-model instance
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred_train = regressor.predict(X_train)
Y_pred_val = regressor.predict(X_val)

# Inverse logarithmic transformation if necessary
if "PRICES" in Log_Features:
	Y_pred_train, Y_pred_val = np.exp(Y_pred_train), np.exp(Y_pred_val)
	Y_train, Y_val = np.exp(Y_train), np.exp(Y_val)

"""
## Show the result
### Check R2 socre
"""
R2 = r2_score(Y_val, Y_pred_val)
st.write(f'R2 score: {R2:.2f}')
"""
### Plot the result
"""
left_column, right_column = st.beta_columns(2)
show_train = left_column.radio(
				'Show the training dataset:', 
				('Yes','No')
				)
show_val = right_column.radio(
				'Show the validation dataset:', 
				('Yes','No')
				)

# default axis range
y_max_train = max([max(Y_train), max(Y_pred_train)])
y_max_val = max([max(Y_val), max(Y_pred_val)])
y_max = int(max([y_max_train, y_max_val])) 

# interactive axis range
left_column, right_column = st.beta_columns(2)
x_min = left_column.number_input('x_min:',value=0,step=1)
x_max = right_column.number_input('x_max:',value=y_max,step=1)
left_column, right_column = st.beta_columns(2)
y_min = left_column.number_input('y_min:',value=0,step=1)
y_max = right_column.number_input('y_max:',value=y_max,step=1)


fig = plt.figure(figsize=(3, 3))
if show_train == 'Yes':
	plt.scatter(Y_train, Y_pred_train,lw=0.1,color="r",label="training data")
if show_val == 'Yes':
	plt.scatter(Y_val, Y_pred_val,lw=0.1,color="b",label="validation data")
plt.xlabel("PRICES",fontsize=8)
plt.ylabel("PRICES of prediction",fontsize=8)
plt.xlim(int(x_min), int(x_max)+5)
plt.ylim(int(y_min), int(y_max)+5)
plt.legend(fontsize=6)
plt.tick_params(labelsize=6)
st.pyplot(fig)




