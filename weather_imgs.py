## Neural network to create weather descriptions
## Ben Conrad
## December 2017
## CMPT 318

import sys
import os
import numpy as np
import pandas as pd
import re
import cv2

from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

## Regex expressions for weather types
# Single weather
clear_re = re.compile(r'.*(clear).*')
cloudy_re = re.compile(r'.*(cloudy).*')
rain_re = re.compile(r'.*(rain|drizzle).*')
fog_re = re.compile(r'.*(fog).*')
snow_re = re.compile(r'.*(snow).*')
ice_re = re.compile(r'.*(ice pellets).*')
thunder_re = re.compile(r'.*(thunderstorm).*')

# Double weather
rain_fog_re = re.compile(r'.*(rain.*fog|drizzle.*fog).*')
snow_fog_re = re.compile(r'.*(snow.*fog).*')
rain_snow_re = re.compile(r'.*(rain.*snow).*')
thunder_rain_re = re.compile(r'.*(thunderstorm.*rain).*')
ice_fog_re = re.compile(r'.*(freezing rain.*fog).*')
rain_ice_re = re.compile(r'.*(rain.*ice|rain.*snow pellets).*')

# Triple Weather
rain_snow_fog_re = re.compile(r'.*(rain.*snow.*fog).*')
snow_ice_fog_re = re.compile(r'.*(snow.*ice pellets.*fog).*')
rain_ice_fog_re = re.compile(r'.*(rain.*snow pellets.*fog|rain.*hail.*fog).*')

########################## Functions ###########################################################################

## Read imgs from file and create dataframe
# Inputs:	
	#directory - string of images directory
def grabImgs(directory):
	imgs = pd.DataFrame(columns=['img', 'Date/Time', 'year', 'month', 'day', 'hour'])
	i = 0
	# Make Dataframe with all imgs
	for f in os.listdir(directory):
		# Create a Series for each img and append to dataframe
		img = pd.Series([cv2.imread(os.path.join(directory,f)), f[7:11], f[11:13], f[13:15], f[15:17]],\
					     index=['img', 'year', 'month', 'day', 'hour'])
		img['img'] = img['img'].flatten().tolist() # Flatten the img matrix into single row array
		img['Date/Time'] = img.apply(lambda row: makeDate([img['year'], img['month'], img['day'], img['hour']]))[0] # Create a date and time col		
		imgs = imgs.append(img, ignore_index = True) # Add the img series into the dataframe of all images

	imgs['img'] = np.array(list(imgs['img']), dtype=np.float) # Cast the img array to float
	return imgs

## Read weather from file and create dataframe
# Inputs:
	#directory - string of weather data directory
def grabWeather(directory):
	weather = pd.DataFrame()
	w_list = []
	# Make dataframe with all weather data
	for f in os.listdir(directory):
		# Read one of the files and append to dataframe
		w = pd.read_csv(os.path.join(directory, f), skiprows= lambda x: x in range(16))
		w_list.append(w)
	weather = pd.concat(w_list)
	return weather

## Fix the weather column value to one of the categorgies
# Inputs:
	# weather - single weather column value
def weatherMatch(weather):
	### Look for compound weather first

	## Triple weather
	snow_ice_fog = snow_ice_fog_re.match(weather)
	if snow_ice_fog:
		return 'snow,ice,fog'

	rain_ice_fog = rain_ice_fog_re.match(weather)
	if rain_ice_fog:
		return 'rain,ice,fog'

	rain_snow_fog = rain_snow_fog_re.match(weather)
	if rain_snow_fog:
		return 'rain,snow,fog'


	## Double weather
	ice_fog = ice_fog_re.match(weather)
	if ice_fog:
		return 'ice,fog'

	rain_fog = rain_fog_re.match(weather)
	if rain_fog:
		return 'rain,fog'

	rain_snow = rain_snow_re.match(weather)
	if rain_snow:
		return 'rain,snow'

	snow_fog = snow_fog_re.match(weather)
	if snow_fog:
		return 'snow,fog'

	thunder_rain = thunder_rain_re.match(weather)
	if thunder_rain:
		return 'thunder,rain'


	rain_ice = rain_ice_re.match(weather)
	if rain_ice:
		return 'rain,ice'


	## Single Weather
	clear = clear_re.match(weather)
	if clear:
		return clear.group(1)

	cloudy = cloudy_re.match(weather)
	if cloudy:
		return cloudy.group(1)

	rain = rain_re.match(weather)
	if rain:
		return 'rain'

	fog = fog_re.match(weather)
	if fog:
		return fog.group(1)

	snow = snow_re.match(weather)
	if snow:
		return snow.group(1)

	ice = ice_re.match(weather)
	if ice:
		return 'ice'

	thunder = thunder_re.match(weather)
	if thunder:
		return 'thunder'

## Create date and time string from date and time parts
# Inputs:
	# dateparts - array with year, month, day and hour ints
def makeDate(dateparts):
	year = str(dateparts[0])
	month = str(dateparts[1])
	day = str(dateparts[2])
	hour = str(dateparts[3])

	return year + '-' + month + '-' + day + ' ' + hour + ':00'

## Create label tuples from the weather descriptions
# Input:
	# row - row from the weather dataframe
def makeLabels(row):
	w = row['Weather']

	if w == 'clear':
		return (1,)

	if w == 'cloudy':
		return (2,)

	if w == 'rain':
		return (3,)

	if w == 'fog':
		return (4,)

	if w == 'snow':
		return (5,)

	if w == 'ice':
		return (6,)

	if w == 'thunder':
		return (7,)



	if w == 'ice,fog':
		return (4,6)

	if w == 'rain,fog':
		return (3,4)

	if w == 'rain,snow':
		return (3,5)
		
	if w == 'snow,fog':
		return (4,5)	

	if w == 'thunder,rain':
		return (3,7)

	if w == 'rain,ice':
		return (3,6)



	if w == 'snow,ice,fog':
		return (4,5,6)	
		
	if w == 'rain,ice,fog':	
		return (3,4,6)

	if w == 'rain,snow,fog':
		return (3,4,5)		
		
## Clean weather dataframe into correct data
# Inputs:
	# weather - weather dataframe
def cleanWeather(weather):
	weather = weather.fillna(method='ffill') # Fill in the empty weather descriptions with the previous description
	weather = weather[pd.notnull(weather['Weather'])] # Remove rows that don't have the weather 
	# Remove all unimportant columns 
	weather = weather.drop(['Data Quality', 'Temp Flag', 'Dew Point Temp Flag', 'Rel Hum Flag', \
							'Wind Dir Flag', 'Wind Spd Flag', 'Visibility Flag', 'Hmdx', 'Wind Dir (10s deg)', \
							'Stn Press Flag', 'Hmdx Flag', 'Wind Chill', 'Wind Chill Flag'], \
							axis=1)

	weather['Weather'] = weather['Weather'].str.lower() # Make weather col all lowercase
	weather['Weather'] = weather['Weather'].apply(weatherMatch) # Fix the weather column values
	

	return weather

## Save csv file with results from testing data
# Inputs:
	# labels - array of predicted labels
	# attributes - dataframe of X_test
	# output_name - name of csv file 
def createOutput(labels, attributes, output_name):
	# Create the output dataframe
	output = pd.DataFrame(attributes.index, columns=['Date/Time'])
	output['Weather'] = pd.DataFrame(labels).apply(lambda x: labels2weather(x), axis=1) # Convert labels to weather descriptions
	output['Filename'] = output.apply(lambda x: date2file(x['Date/Time']), axis=1) # Make filename from date

	output.to_csv(output_name, index=False)
	
## Create weather descriptions from muli-label values
# Inputs:
	# label - binary array of multi-label values
def labels2weather(label):
	label = label.as_matrix()
	weather = ''
	
	# Concat weather descriptors if the label is true
	if label[0] == 1:
		weather = weather + 'clear,'

	if label[1] == 1:
		weather = weather + 'cloudy,'

	if label[2] == 1:
		weather = weather + 'rain,'

	if label[3] == 1:
		weather = weather + 'fog,'

	if label[4] == 1:
		weather = weather + 'snow,'

	try:
		if label[5] == 1:
			weather = weather + 'ice pellets,'
	except:
		pass

	try:	
		if label[6] == 1:
			weather = weather + 'thunder storms,'
	except:
		pass

	if weather == '':
		return weather
	else:
		return weather[:-1]

## Get img filename from date
# Input:
	# date - a Date/Time column value 
def date2file(date):
	return 'katkam-' + date[0:4] + date[5:7] + date[8:10] + date[11:13] + '0000'

################################################################################################################


def main(img_directory, weather_directory, output):
	imgs = grabImgs(img_directory) # Make img dataframe

	weather = grabWeather(weather_directory) # Make weather dataframe
	weather = cleanWeather(weather) # Clean weather dataframe

	# Merge imgs and weather together
	data = pd.merge(imgs, weather, how='inner', left_on='Date/Time', right_on='Date/Time').drop(['year','month','day'], axis=1)

	# Add a multi-label column to denote weather observations
	data['labels'] = data.apply(lambda x: makeLabels(x), axis=1)

	data = data.set_index('Date/Time')

	# Split Data
	# Observations
	X = data[['img',\
			  'Temp (°C)',\
			  'Dew Point Temp (°C)',\
			  'Rel Hum (%)',\
			  'Wind Spd (km/h)',\
			  'Visibility (km)',\
			  'Stn Press (kPa)',\
			  'Month',\
			  'hour']]

	# Results
	Y = data['labels']
	Y = MultiLabelBinarizer().fit_transform(Y)

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y,)

	# Create Model
	model = make_pipeline(#PCA(4),\
						  StandardScaler(),\
						  OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=(500,), activation='relu', max_iter=3000,\
						  								    learning_rate_init=0.001, learning_rate='adaptive', solver='adam',\
						  									alpha=0.001, warm_start=True, beta_1=0.95, beta_2=0.95, epsilon=1e-8)),
						  
						  #OneVsRestClassifier(DecisionTreeClassifier()),
						  #OneVsRestClassifier(SVC(kernel='rbf', C=1)),
						  #OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3)),
						  )


	# Train and Test Model
	model.fit(X_train, Y_train)

	Y_predicted = model.predict(X_test)
	print(accuracy_score(Y_test, Y_predicted))

	# Save results to CSV
	createOutput(Y_predicted, X_test, output)





if __name__=='__main__':
	img_directory = sys.argv[1]
	weather_directory = sys.argv[2]
	output = sys.argv[3]
	main(img_directory, weather_directory, output)