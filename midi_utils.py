import glob
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
from lib.MidiOutFile import MidiOutFile
from utils import midiread

# Picks .mid files, preprocess them and outputs a list of songs
def midi_to_events_roll(directory):
	path = directory + '/*.mid'
	files = glob.glob(path)
	notes_range = find_notes_range(directory)
	left_hand_top = 64
	max_vel = 128.
	notes_length = notes_range['M'] - notes_range['m'] + 1
	vector_len = notes_length + 2 + 2 + 1	# n bit per nota + 2 bit per velocity + 2 bit per durata nota + tempo attesa per prossima nota
	events_set = []
	for file in files:
		try:
			m = midiread(file)
		except Exception:
			continue
		i = 0
		song = []
		#faccio sorting prima sulla nota, poi sul tempo di inizio della nota
		events = sorted(m.notes, key=lambda x: x[0])
		events = sorted(events, key=lambda x: x[1])
		for j in xrange(len(events) + 1):
			if j != len(events) and events[i][1] == events[j][1]:	#unisco le note che partono nello stesso istante; se non sono alla fine, continuo a unificare note
					continue
			a = [0] * vector_len
			hands = ([] , [])
			notes = events[i:j]
			for p in notes:
				a[p[0] - notes_range['m']] = 1	#codifico le note premute insieme nel vettore corrente
				if p[0] < left_hand_top: 	#cosi' dico che le note suonate nella parte sinistra della tastiera sono suonate con la mano sx.
					hands[0].append(p)
				else:
					hands[1].append(p)
			for idx in range(0,2):
				if len(hands[idx]):
					vel = [l[3] for l in hands[idx]]
					span = [l[2]- l[1] for l in hands[idx]]
					a[notes_length+idx] = float("%.3f" % ((sum(vel)/len(vel))/max_vel))
					a[notes_length+idx+2] = float("%.3f" % (sum(span)/len(span)))
			if j < len(events):
				a[-1] = events[j][1] - events[i][1]
			else:
				a[-1] = float("%.2f" % (max([n[2] for n in events]) - events[i][1])) #se sono all'ultimo accordo, indico come tempo di attesa per la prossima nota il tempo in cui smette di suonare l'ultima nota
			i = j
			song.append(a)

		events_set.append(song)

	return events_set, notes_range

def find_notes_range(directory):
	path = directory + '/*.mid'
	files = glob.glob(path)
	notes_range = {'m': 128, 'M': 0}
	for file in files:
		try:
			m = midiread(file)
		except Exception:
			continue
		for note in m.notes:
			if note[0] < notes_range['m']:
				notes_range['m'] = note[0]
			if note[0] > notes_range['M']:
				notes_range['M'] = note[0]
	return notes_range

def create_pickle(directory, filename, valid_perc=0):
	l, offset = midi_to_events_roll(directory)
	if not valid_perc:
		d = {'train' : l, 'offset': offset}
	else:
		valid_idx = int(len(l)*valid_perc)
		l_train = l[:-valid_idx]
		l_valid = l[-valid_idx:]
		d = {'train' : l_train, 'valid' : l_valid, 'offset': offset}

	with open(filename + '.pickle', 'wb') as handle:
		pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

def sample(preds, temperature=1.0):
	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds) / temperature
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	probas = np.random.binomial(1, preds)
	return probas

def plot(filename, probs, lowest_note):
	x = []
	y = []
	alphas = []
	for time_index, timestep in enumerate(probs):
		for note_index, prob in enumerate(timestep):
			x.append(time_index)
			y.append(note_index + lowest_note)
			alphas.append(prob)

	colors = np.zeros((len(x),4))
	colors[:, 3] = alphas
	plt.scatter(x, y, c=colors, marker='_')
	plt.savefig(filename+'.png')
	plt.clf()