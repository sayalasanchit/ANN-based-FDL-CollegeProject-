from keras.models import Sequential
from keras.layers import Dense, Activation
optm='adam'

# Model architecture
def singlular_ann_model():
	model_single=Sequential()
	model_single.add(Dense(40, input_shape=(9, )))
	model_single.add(Activation('tanh'))
	model_single.add(Dense(1))
	model_single.add(Activation('relu'))
	model_single.compile(loss='mean_squared_error', optimizer=optm, metrics=['loss'])
	return model_single