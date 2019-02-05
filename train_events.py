import os
import pickle
import numpy as np
import keras
import tensorflow as tf
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
from keras import optimizers
from keras import backend as K
from models import GenerativeEventsLSTM, GenerativeLSTM, checkpoint_callback

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
def main():
	'''
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	set_session(tf.Session(config=config))
	'''
	filename = 'Nottingham'
	filepath = '{}.pickle'.format(filename)

	with open(filepath, 'rb') as file:
		dataset = pickle.load(file)

	highest_note = dataset['offset']['M']
	lowest_note = dataset['offset']['m']
	sequence_length = 50
	notes_dim = highest_note - lowest_note + 1	#per Nottingham 53
	vel_dim = 2
	times_dim = 3
	input_dim = notes_dim + vel_dim + times_dim	#per Nottingham 58
	validation_percent = .25

	only_notes = False

	if only_notes:
		training_events = [event[:notes_dim] for song in dataset['train'] for event in song]
	else:
		training_events = [event for song in dataset['train'] for event in song]

	X = []
	y = []
	for i in range(int(len(training_events)) - 1):
		if i + sequence_length >= len(training_events):
			break
		X.append(training_events[i:i+sequence_length])
		y.append(training_events[i+sequence_length])

	y1 = [o[:notes_dim] for o in y]
	y2 = [o[notes_dim:notes_dim+vel_dim] for o in y]
	y3 = [o[notes_dim+vel_dim:] for o in y]

	if not only_notes:
		model = GenerativeEventsLSTM(notes_dim=notes_dim, vel_dim=vel_dim, times_dim=times_dim, sequence_length=sequence_length, hidden_size=128, dropout=0.25)
		model.compile(optimizer=optimizers.adam(), loss=['binary_crossentropy', 'mean_squared_error', 'mean_squared_error'])
		print(model.summary())
		checkpoint = checkpoint_callback(model)
		result = model.fit(np.asarray(X), [np.asarray(y1), np.asarray(y2), np.asarray(y3)], validation_split=validation_percent, epochs=200, batch_size=128, callbacks=[checkpoint])
	else:
		X = X[:int(len(X)/2)]
		y1 = y1[:int(len(y1)/2)]
		model = GenerativeLSTM(input_dim=notes_dim, sequence_length=sequence_length, hidden_size=128, dropout=0.25)
		model.compile(optimizer=optimizers.adam(), loss=['binary_crossentropy'])
		checkpoint = checkpoint_callback(model)
		result = model.fit(np.asarray(X), np.asarray(y1), validation_split=validation_percent, epochs=100, batch_size=128, callbacks=[checkpoint])

if __name__ == '__main__':
	main()
