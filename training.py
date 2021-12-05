# Importing libraries
import pandas as pd
import numpy as np
from modular_functions import fault_type, modular_predict, modular_train
from singlular_ann_archtecture import singlular_ann_model
from modular_ann_archtecture import modular_ann_model
from sklearn import preprocessing

model_types=['phase2ground', 'phase2phase', 'doublephase2ground', 'threephase']

def load_data(filename):
	# Importing dataset
	df=pd.read_excel(filename)

	lables=df.values[:, :7]
	input_data=df.values[:, 7: 16]
	target=df.values[:, -1]

	# Preprocessing
	scaler=preprocessing.StandardScaler().fit(input_data)
	input_data=scaler.transform(input_data)
	return labels, input_data, target


def train_sigular(input_data, target):
	# Singular model
	model_single=singlular_ann_model()
	# Singular model training
	hist=model_single.fit(input_data, target, epochs=300, batch_size=10, validation_split=0.2)
	return hist


def train_modular(labels, input_data, target, model_types):
	# Modular model
	model_modular=modular_ann_model(model_types)
	# Modular model training
	hist_c, hist_r=modular_train(input_data, target, labels, model_modular["classifier"], model_modular["regressor"], model_types)
	hist={"classifier": hist_c, "regressor": hist_r}
	return hist


if __name__=="__main__":
	# Loading the data
	labels, input_data, target=load_data("generated_data.xls")

	# Training both models
	hist_singular=train_sigular(input_data, target)
	hist_modular=train_modular(labels, input_data, target, model_types)

	# Final validation losses
	final_loss_single=hist_singular['val_loss'][-1]
	print(f"The final loss (validation) for singular ANN model is: {final_loss_single}")

	final_losses_modular={}
	print("The final losses (validation) for modular ANN model are:")
	for model_type in model_types:
		final_losses_modular[model_type]=hist_modular["regressor"][model_type]['val_loss'][-1]
		print(f"Loss for {model_type} is: {final_losses_modular[model_type]}")
