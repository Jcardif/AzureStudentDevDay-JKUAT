#  Download the data using keras
from keras.datasets import boston_housing
(x_train, y_train),(x_test,y_test)=boston_housing.load_data()

#This directly downoads the data into the Python environment for use

# Import the necessary modules
import numpy as numpy
from keras.models import Sequential
from keras.layers import Dense, Activation

#Extract the last 100 rows from the training data to create validation datasets
x_val=x_train[300:,]
y_val=y_train[300:,]

#Define the Model Arrchitecture
model=Sequential()
model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
model.add(Dense(6,kernel_initializer='normal',activation='relu'))
model.add(Dense(1,kernel_initializer='normal',activation='relu'))

#Compile the model
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mean_absolute_percentage_error'])

#Train the Model
model.fit(x_train,y_train, batch_size=32, epochs=100000,validation_data=(x_val,y_val))

#Evaluate the model
results=model.evaluate(x_test,y_test)

for i in range(len(model.metrics_names)):
    print(model.metrics_names[i]," : ",results[i])