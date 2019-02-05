import keras
from keras import Model
from keras.layers import Dense, Bidirectional, LSTM, Input, multiply
from keras.layers.core import *
from keras.models import *
import matplotlib.pyplot as plt

def GenerativeLSTM(input_dim, sequence_length, hidden_size, dropout):
	input_data = Input(shape=(sequence_length, input_dim))
	x = Bidirectional(LSTM(hidden_size, return_sequences=True, dropout=dropout))(input_data)
	attention_mul = attention_3d_block(x, sequence_length)
	x = Bidirectional(LSTM(hidden_size, dropout=dropout))(attention_mul)
	output = Dense(input_dim, activation="sigmoid")(x)
	model = Model(input_data, output)
	return model

def GenerativeEventsLSTM(notes_dim, vel_dim, times_dim, sequence_length, hidden_size, dropout):
	input_data = Input(shape=(sequence_length, notes_dim+vel_dim+times_dim))
	x = Bidirectional(LSTM(hidden_size, return_sequences=True, dropout=dropout))(input_data)
	attention_mul = attention_3d_block(x, sequence_length)
	x = Bidirectional(LSTM(hidden_size, dropout=dropout))(attention_mul)
	output1 = Dense(notes_dim, activation="sigmoid")(x)
	output2 = Dense(vel_dim, activation="sigmoid")(x)
	output3 = Dense(times_dim, activation="relu")(x)
	model = Model(input_data, [output1, output2, output3])
	return model

def attention_3d_block(inputs, time_steps):
	input_dim = int(inputs.shape[2])
	a = Permute((2, 1))(inputs)
	a = Dense(time_steps, activation='softmax')(a)
	a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
	a = RepeatVector(input_dim)(a)
	a_probs = Permute((2, 1), name='attention_vec')(a)
	output_attention_mul = multiply([inputs, a_probs], name='att_output')
	return output_attention_mul

class checkpoint_callback(keras.callbacks.Callback):

		def __init__(self, model):
			 self.model_to_save = model

		def on_epoch_end(self, epoch, logs=None):
			self.model_to_save.save('model_at_epoch_%d.h5' % epoch)