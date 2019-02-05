# GenerativeLSTM
A simple recurrent model for piano music generation.

Files under the /lib directory were developed by Nicolas Boulanger-Lewandowski @ University of Montreal.

If you want to try it out with your own music selection, run the following commands:
'''
from midi_utils import create_pickle as cp
cp('path/to/dir', 'pickle_name')
'''
This will create a .pickle file you can then use to train your model on. Just change the 'filename' variable to the created 'pickle_name' in train_events.py. To then generate new tracks, simply run generate.py.
