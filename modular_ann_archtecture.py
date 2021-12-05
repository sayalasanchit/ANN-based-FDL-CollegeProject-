# Importing libraries
from keras.models import Sequential
from keras.layers import Dense, Activation
optm='adam'

# Model architecture
def modular_ann_model(model_types):
	# Fault type classifiers
	model_classifier=Sequential()
	model_classifier.add(Dense(10, input_shape=(9, )))
	model_classifier.add(Activation('tanh'))
	model_classifier.add(Dense(7))
	model_classifier.add(Activation('sigmoid'))
	model_classifier.compile(loss='categorical_crossentropy', optimizer=optm, metrics=['accuracy'])

	# Fault location regressors
	model_modular={}
	hidden_layer_neurons={
	    'phase2ground':30, 
	    'phase2phase':30, 
	    'doublephase2ground':20, 
	    'threephase':5
	}
	for model_type in model_types:
		model_modular[model_type]=Sequential()
		model_modular[model_type].add(Dense(hidden_layer_neurons[model_type], input_shape=(9, )))
		model_modular[model_type].add(Activation('tanh'))
		model_modular[model_type].add(Dense(1))
		model_modular[model_type].add(Activation('relu'))
		model_modular[model_type].compile(loss='mean_squared_error', optimizer=optm, metrics=['loss'])

	model={"classifier": model_classifier, "regressor":model_modular}
	return model