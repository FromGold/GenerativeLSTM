import pickle
import numpy as np
import keras
from keras import optimizers
from keras import backend as K
from random import randint
from models import GenerativeEventsLSTM

#midi library
from utils import midiwrite
from midi_utils import sample, plot

def main():
	filename = 'Nottingham'
	filepath = '{}.pickle'.format(filename)
	song_len = 500
	vel_dim = 2
	times_dim = 3
	sequence_length = 50

	with open(filepath, 'rb') as file:
		dataset = pickle.load(file)

	highest_note = dataset['offset']['M']
	lowest_note = dataset['offset']['m']
	notes_dim = highest_note - lowest_note + 1
	total_dim = notes_dim + vel_dim + times_dim

	test_notes = [timestep for song in dataset['valid'] for timestep in song]

	model = GenerativeEventsLSTM(notes_dim=notes_dim, vel_dim=vel_dim, times_dim=times_dim, sequence_length=sequence_length, hidden_size=128, dropout=0.25)
	model.compile(optimizer=optimizers.adam(), loss=['binary_crossentropy', 'mean_squared_error', 'mean_squared_error'])
	
	model.load_weights(filename + '.h5')
	random_int = randint(0, len(test_notes) - sequence_length - 1)

	input_sequence = np.asarray(test_notes[random_int:random_int + sequence_length])

	#lista di note generate
	generated_notes = []
	probabilities = []
	for i in range(song_len):
		out = model.predict(np.reshape(input_sequence, (1, sequence_length, total_dim)))
		out1 = out[0][0]
		out2 = out[1][0]
		out3 = out[2][0]
		probabilities.append(out1.tolist())
		out1 = sample(out1, temperature=1.5)
		joint = np.concatenate((out1, out2, out3), axis=0)
		generated_notes.append(joint.tolist())
		input_sequence = np.insert(input_sequence, sequence_length, joint, axis=0)[1:]

	plot(filename, probabilities, lowest_note)
	midiwrite(filename, generated_notes, (lowest_note, highest_note))

if __name__ == '__main__':
	main()