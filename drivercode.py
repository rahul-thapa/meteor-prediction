import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Path

meteor_path = 'meteorite-landings.csv'

meteor_data = pd.read_csv(meteor_path)

#********************************************************************
# MAPPING the data on a world map using the co-ordinates
import matplotlib.pyplot as plt

# NEED HELP HERE



#*********************************************************************

meteor_data = meteor_data.dropna(axis = 0)

# I am trying to predict the location of meteor landing

y = meteor_data[['reclat','reclong']]

features = ['mass','year']

X = meteor_data[features]


from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X,y, random_state=1)

# I am using the decision tree regressor model here because of its simplicity

meteor_model = DecisionTreeRegressor(random_state = 1)

meteor_model.fit(train_X, train_y)

val_predictions = meteor_model.predict(X.head())

#Making predictions for this data:

print(X.head())
print("The predictions are:")
print(val_predictions)


