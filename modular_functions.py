import numpy as np

def fault_type(A1, B1, C1, A2, B2, C2, N):
  if A1==0 and A2==0 and B1==0 and B2==0 and C1==0 and C2==0 and N==0:
    return 'nofault'
  if N:
    if (A1 and B1) or (A2 and B2) or (B1 and C1) or (B2 and C2) or (C1 and A1) or (C2 and A2):
      return 'doublephase2ground'
    if A1 or A2 or B1 or B2 or C1 or C2:
      return 'phase2ground'
  else:
    if (A1 and B1 and C1) or (A2 and B2 and C2):
      return 'threephase'
    if (A1 and B1) or (A2 and B2) or (B1 and C1) or (B2 and C2) or (C1 and A1) or (C2 and A2):
      return 'phase2phase'


def modular_train(input_data, output_data, labels, model_classifier, model_modular, model_types, epoch_c=300, epoch_r=300, batch=10, val_split=0.2):
  # 1. Training classifier

  hist_c=model_classifier.fit(input_data, labels, epochs=epoch_c, batch_size=batch, validation_split=val_split)

  # 2. Training regressor

  # Seperating the data according to type of fault
  labelled_input_data, labelled_output_data={}, {}
  for model_type in model_types:
    labelled_input_data[model_type]=[]
    labelled_output_data[model_type]=[]
  for i in range(input_data.shape[0]):
    fault=fault_type(*labels[i]) # Finding the type of data
    if fault=='nofault':
      continue
    labelled_input_data[fault].append(input_data[i])
    labelled_output_data[fault].append(output_data[i])

  # Making arrays for each list
  for model_type in model_types:
    labelled_input_data[model_type]=np.array(labelled_input_data)
    labelled_output_data[model_type]=np.array(labelled_output_data)
  hist_r={}
  for model_type in model_types:
    hist_r[model_type]=model_modular[model_type].fit(labelled_input_data[model_type],
                                                      labelled_output_data[model_type],
                                                      epochs=epoch_r,
                                                      batch_size=batch,
                                                      validation_split=val_split)
  return hist_c, hist_r


def modular_predict(data, model_classifier, model_modular):
  # 1. Predicting the class

  classes_pred=[1 if x>=0.5 else 0 for x in model_classifier.predict(data)]

  # 2. Predicting the distance

  fault=fault_type(*classes_pred)
  if fault=='nofault':
    return -1
  pred_distance=model_modular[fault].predict(data)
  return pred_distance