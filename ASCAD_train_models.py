import os
import os.path
import sys
import h5py
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, AveragePooling1D
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.models import load_model

def check_file_exists(file_path):
	file_path = os.path.normpath(file_path)
	if os.path.exists(file_path) == False:
		print("Error: provided file path '%s' does not exist!" % file_path)
		sys.exit(-1)
	return

#### MLP Best model (6 layers of 200 units)
def mlp_best(node=200,layer_nb=6):
	model = Sequential()
	model.add(Dense(node, input_dim=1400, activation='relu'))
	for i in range(layer_nb-2):
		model.add(Dense(node, activation='relu'))
	model.add(Dense(256, activation='softmax'))
	optimizer = RMSprop(lr=0.00001)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model

### CNN Best model
def cnn_best(classes=256):
	# From VGG16 design
	input_shape = (700,1)
	img_input = Input(shape=input_shape)
	# Block 1
	x = Conv1D(64, 11, activation='relu', padding='same', name='block1_conv1')(img_input)
	x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
	# Block 2
	x = Conv1D(128, 11, activation='relu', padding='same', name='block2_conv1')(x)
	x = AveragePooling1D(2, strides=2, name='block2_pool')(x)
	# Block 3
	x = Conv1D(256, 11, activation='relu', padding='same', name='block3_conv1')(x)
	x = AveragePooling1D(2, strides=2, name='block3_pool')(x)
	# Block 4
	x = Conv1D(512, 11, activation='relu', padding='same', name='block4_conv1')(x)
	x = AveragePooling1D(2, strides=2, name='block4_pool')(x)
	# Block 5
	x = Conv1D(512, 11, activation='relu', padding='same', name='block5_conv1')(x)
	x = AveragePooling1D(2, strides=2, name='block5_pool')(x)
	# Classification block
	x = Flatten(name='flatten')(x)
	x = Dense(4096, activation='relu', name='fc1')(x)
	x = Dense(4096, activation='relu', name='fc2')(x)
	x = Dense(classes, activation='softmax', name='predictions')(x)

	inputs = img_input
	# Create model.
	model = Model(inputs, x, name='cnn_best')
	optimizer = RMSprop(lr=0.00001)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model

### CNN Best model
def cnn_best2(classes=256):
	# From VGG16 design
	input_shape = (1400,1)
	img_input = Input(shape=input_shape)
	# Block 1
	x = Conv1D(64, 11, strides=2, activation='relu', padding='same', name='block1_conv1')(img_input)
	x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
	# Block 2
	x = Conv1D(128, 11, activation='relu', padding='same', name='block2_conv1')(x)
	x = AveragePooling1D(2, strides=2, name='block2_pool')(x)
	# Block 3
	x = Conv1D(256, 11, activation='relu', padding='same', name='block3_conv1')(x)
	x = AveragePooling1D(2, strides=2, name='block3_pool')(x)
	# Block 4
	x = Conv1D(512, 11, activation='relu', padding='same', name='block4_conv1')(x)
	x = AveragePooling1D(2, strides=2, name='block4_pool')(x)
	# Block 5
	x = Conv1D(512, 11, activation='relu', padding='same', name='block5_conv1')(x)
	x = AveragePooling1D(2, strides=2, name='block5_pool')(x)
	# Classification block
	x = Flatten(name='flatten')(x)
	x = Dense(4096, activation='relu', name='fc1')(x)
	x = Dense(4096, activation='relu', name='fc2')(x)
	x = Dense(classes, activation='softmax', name='predictions')(x)

	inputs = img_input
	# Create model.
	model = Model(inputs, x, name='cnn_best2')
	optimizer = RMSprop(lr=0.00001)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model

def load_sca_model(model_file):
	check_file_exists(model_file)
	try:
        	model = load_model(model_file)
	except:
		print("Error: can't load Keras model file '%s'" % model_file)
		sys.exit(-1)
	return model

#### ASCAD helper to load profiling and attack data (traces and labels)
# Loads the profiling and attack datasets from the ASCAD
# database
def load_ascad(ascad_database_file, load_metadata=False):
	check_file_exists(ascad_database_file)
	# Open the ASCAD database HDF5 for reading
	try:
		in_file  = h5py.File(ascad_database_file, "r")
	except:
		print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
		sys.exit(-1)
	# Load profiling traces
	X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.int8)
	# Load profiling labels
	Y_profiling = np.array(in_file['Profiling_traces/labels'])
	# Load attacking traces
	X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.int8)
	# Load attacking labels
	Y_attack = np.array(in_file['Attack_traces/labels'])
	if load_metadata == False:
		return (X_profiling, Y_profiling), (X_attack, Y_attack)
	else:
		return (X_profiling, Y_profiling), (X_attack, Y_attack), (in_file['Profiling_traces/metadata'], in_file['Attack_traces/metadata'])

#### Training high level function
def train_model(X_profiling, Y_profiling, model, save_file_name, epochs=150, batch_size=100):
	check_file_exists(os.path.dirname(save_file_name))
	# Save model every epoch
	save_model = ModelCheckpoint(save_file_name)
	callbacks=[save_model]
	# Get the input layer shape
	input_layer_shape = model.get_layer(index=0).input_shape
	# Sanity check
	if input_layer_shape[1] != len(X_profiling[0]):
		print("Error: model input shape %d instead of %d is not expected ..." % (input_layer_shape[1], len(X_profiling[0])))
		sys.exit(-1)
	# Adapt the data shape according our model input
	if len(input_layer_shape) == 2:
		# This is a MLP
		Reshaped_X_profiling = X_profiling
	elif len(input_layer_shape) == 3:
		# This is a CNN: expand the dimensions
		Reshaped_X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
	else:
		print("Error: model input shape length %d is not expected ..." % len(input_layer_shape))
		sys.exit(-1)
	
	history = model.fit(x=Reshaped_X_profiling, y=to_categorical(Y_profiling, num_classes=256), batch_size=batch_size, verbose = 1, epochs=epochs, callbacks=callbacks)
	return history

def read_parameters_from_file(param_filename):
	#read parameters for the train_model and load_ascad functions
	#TODO: sanity checks on parameters
	param_file = open(param_filename,"r")

	#TODO: replace eval() by ast.linear_eval()
	my_parameters= eval(param_file.read())

	ascad_database = my_parameters["ascad_database"]
	training_model = my_parameters["training_model"]
	network_type = my_parameters["network_type"]
	epochs = my_parameters["epochs"]
	batch_size = my_parameters["batch_size"]
	return ascad_database, training_model, network_type, epochs, batch_size


if __name__ == "__main__":
	if len(sys.argv)!=2:
		#default parameters values
		ascad_database = "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5"
		#MLP training
		network_type = "mlp"
		training_model = "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_trained_models/my_mlp_best_desync0_epochs75_batchsize200.h5"

		#CNN training
		#network_type = "cnn"
		#training_model = "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_trained_models/my_cnn_best_desync0_epochs75_batchsize200.h5"

		#CNN training
		#network_type = "cnn2"
		#training_model = "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_trained_models/my_cnn_best_desync0_epochs75_batchsize200.h5"

		epochs = 75
		batch_size = 200
	else:
		#get parameters from user input
		ascad_database, training_model, network_type, epochs, batch_size = read_parameters_from_file(sys.argv[1])
	#load traces
	(X_profiling, Y_profiling), (X_attack, Y_attack) = load_ascad(ascad_database)

	#get network type
	if(network_type=="mlp"):
		best_model = mlp_best()
	elif(network_type=="cnn"):
		best_model = cnn_best()		
	elif(network_type=="cnn2"):
		best_model = cnn_best2()
	else: #display an error and abort
		print("Error: no topology found for network '%s' ..." % network_type)
		sys.exit(-1);

	### training
	train_model(X_profiling, Y_profiling, best_model, training_model, epochs, batch_size)

