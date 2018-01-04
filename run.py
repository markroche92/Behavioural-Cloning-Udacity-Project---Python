from load import *
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

''' Selection of architecture '''
architecture = 'NVIDIA'

dropout_factor = 0.2
learning_rate = 0.0001

''' Definition of model architecture '''
# Normalization and cropping layers
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape= (160, 320, 3)))
model.add(Cropping2D(cropping = ((70, 25), (0, 0))))

if architecture == 'NVIDIA':
	# Implementation of NVIDIA architecture via Keras
	model.add(Convolution2D(24, 5, 5, subsample = (2, 2), activation = 'relu'))
	model.add(Dropout(dropout_factor))
	model.add(Convolution2D(36, 5, 5, subsample = (2, 2), activation = 'relu'))
	model.add(Dropout(dropout_factor))
	model.add(Convolution2D(48, 5, 5, subsample = (2, 2), activation = 'relu'))
	model.add(Dropout(dropout_factor))
	model.add(Convolution2D(64, 3, 3, activation = 'relu'))
	model.add(Dropout(dropout_factor))
	model.add(Convolution2D(64, 3, 3, activation = 'relu'))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(1))

elif architecture == 'LENET':
	# Implementation of LeNet via Keras
	model.add(Convolution2D(6, 5, 5, activation = 'relu'))
	model.add(MaxPooling2D())
	model.add(Convolution2D(6, 5, 5, activation = 'relu'))
	model.add(MaxPooling2D())
	model.add(Flatten())
	model.add(Dense(120))
	model.add(Dense(84))
	model.add(Dense(1))

#model.compile(loss = 'mse', optimizer=Adam(lr = learning_rate))
model.compile(loss = 'mse', optimizer=Adam(lr = learning_rate))

model.fit_generator(train_data, steps_per_epoch = num_batch_train, 
					max_queue_size = 14, epochs = 14, 
					validation_data = valid_data, validation_steps = num_batch_valid,
					verbose = 1, shuffle = True)

model.save('model.h5')
