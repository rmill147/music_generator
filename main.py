import random
import numpy as np
# from matplotlib import pylab
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
from keras import Sequential
from keras.src.layers import SimpleRNN, Dense
from numpy.matlib import repmat
import tensorflow as tf
import pretty_midi

window_length = 12  # number of notes used to predict the next note
# training parameters

batch_size = None   # size of each batch (None is default)
song_length = 15
loss_threshold = 0.001


# My attempt at fixing circular import issues...
class Generator:
    def __init__(self, copies=0):
        self._copies = int(copies)

        # getter method
    def get_copies(self):
        return int(self._copies)
        print(f"copies: {copies}")
        print(type(copies))

        # setter method
    def set_copies(self, x):
        self._copies = x

    def __init__(self, num_nodes=0):
        self._num_nodes = num_nodes

        # getter method
    def get_num_nodes(self):
        return self._num_nodes

        # setter method
    def set_num_nodes(self, x):
        self._num_nodes = x

    def __init__(self, epoch_num=0):
        self._epoch_num = epoch_num

        # getter method
    def get_epoch_num(self):
        return self._epoch_num

        # setter method
    def set_epoch_num(self, x):
        self._epoch_num = x


# Generate random notes to a song and see if the RNN will memorize
def random_song_generator(notes_number):
    output = []
    possible_notes = list(range(1,127))
    for _ in range(notes_number):
        new_note = random.choice(possible_notes)
        output.append(new_note)

    return output


# Take in a midi file
def midi_to_notes(midi_file: str):
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = []

    for note in instrument.notes:
        notes.append(note.pitch)

    print("NOTES")
    print(notes)

    return notes

def shorten_song(song,length):
    output = []
    for i in range(length):
        output.append(song[i])
    return output


# RM: Custom callback for early stopping
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') is not None and logs.get('loss') <= loss_threshold:
            print(f"\nReached {loss_threshold} loss, stopping training!!")
            self.model.stop_training = True


# RM: make PM object from given notes
def notes_to_midi(input, out_file: str):
    # Create a PrettyMIDI object
    output = pretty_midi.PrettyMIDI()
    # Create an Instrument instance for a piano instrument
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    instrument = pretty_midi.Instrument(program=piano_program)

    start_t = 0
    end_t = 0.5

    # Iterate over note names, which will be converted to note number later
    for x in input:
        # Create a Note instance
        note = pretty_midi.Note(velocity=100, pitch=x, start=start_t, end=end_t)
        # Add it to instrument
        instrument.notes.append(note)
        start_t = start_t + 0.5
        end_t = end_t + 0.5


    # Add the instrument to the PrettyMIDI object
    output.instruments.append(instrument)

    output.write(out_file)


def dataset_from_song(song, copies, window_length):
    # repeat song "copies" times
    songs = repmat(song, 1, copies)[0] # number of notes, 1 row, each row repeats number of copies
    # number of windows used ; used to determine what gets added to training set... manipulate to specify number of
    # notes to predict?... (8,1,3) - 12
    num_windows = len(songs) - window_length
    # num_windows = song_length

    x_train,y_train = [],[]
    for i in range(num_windows):
        # get a "window_length" number of notes
        x_train.append(songs[i:i+window_length])
        # get the note after the window
        y_train.append(songs[i+window_length])

    # convert to numpy arrays
    x_train = np.array(x_train,dtype='float32')
    x_train = np.expand_dims(x_train,axis=-1)
    y_train = np.array(y_train,dtype='float32')
    y_train = np.expand_dims(y_train,axis=-1)

    return x_train,y_train

# selecting a song
# sample_file =
# midi_file = midi_to_notes(song)

# song going in
# song_midi = midi_file
# print("Input song:")
# print(song)
# print("Input song length:")
# print(len(song))

# generate a dataset from copies of the song
# x_train,y_train = dataset_from_song(this.song,get_copies,window_length)

# RM: write out input song
# out_file = 'input.mid'
# in_pm = notes_to_midi(song_midi, out_file=out_file)


# Make the actual model and train it
def make_and_train(x_train, y_train, nn_nodes, epochs):

    # specify the architecture of the neural network
    model = Sequential()
    model.add(SimpleRNN(nn_nodes,activation='relu'))
    model.add(Dense(1, activation=None))
    # set up the neural network
    model.compile(
        loss='MeanSquaredError',
        optimizer='Adam',
        metrics=None)
    # use this to save the best weights found so far
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.0001, patience=50,restore_best_weights=True)

    my_callback = myCallback()

    # train the neural network from data
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,epochs=epochs,callbacks=[my_callback,callback])

    print("finished training:")
    model.evaluate(x_train, y_train)

    predictions = model.predict(x_train).round()
    correct = sum(predictions == y_train)
    N = len(predictions)
    print(f"\nTrain set accuracy: {100*correct/N} ({correct}/{N})")
    #print("train set accuracy: %.4f%% (%d/%d)" % (100*correct/N,correct,N))

    return model


# model = make_and_train(x_train, y_train)


# test our model
def do_test(model, test_song, window_length, song_length):
    # shift the scale up by half steps
    print("Test song:")
    print(test_song)

    print(f"length of test song: {len(test_song)}")
    # copies_test = 3

    # reverse the scale (if you want)
    # test_song = song[-1::-1]

    copies_test = round((float(song_length) + 12.0) / float(len(test_song)))
    x_test, y_test = dataset_from_song(test_song,copies_test, window_length)
    predictions = model.predict(x_test).round()

    # RM: Convert predictions to normal array for PM
    pred_list = predictions.reshape(-1)
    pred_list = pred_list.astype(int)

    correct = sum(predictions == y_test)
    N = len(predictions)
    test_accuracy = (correct/N)
    print(f"\nTest set accuracy: {100*correct/N}% ({correct}/{N})")
    #print(" test set accuracy: %.4f%% (%d/%d)" % (100*correct/N,correct,N))

    return pred_list, test_accuracy


# pred_list, test_accuracy = do_test(model, test_song)
#
# # Write out output song
#
# out_file2 = 'output.mid'
# out_pm = notes_to_midi(pred_list, out_file=out_file2)
#
# print("Output song: ")
# print(pred_list)

# 2. TODO: Generate a song ... can it learn a pattern to make a song?
def predict_notes(model, pred_list):
    pred_list_c = np.asarray(pred_list)
    pred_list_c = np.reshape(pred_list_c, (1, 1767, 1))

    predictions = model.predict(pred_list_c).round()

    pitch = predictions[0]
    print("Guessed pitch")
    print(pitch)

    return int(pitch)

def generate_notes(pred_list):
    generated_song = []

    # copy over existing predictions from pred_list
    for i in range(0, len(pred_list)):
        generated_song.append(pred_list[i])
        # generated_song[i] = pred_list[i]

    # Add/predict specified number of notes after
    for x in range(song_length):
    #     note = predict_notes(song_length,x_train, pred_list)
        note = predict_notes(pred_list)
        generated_song.append(note)
        # I want to update the pred list so that the first note is moved from spot 0 to the back.
        selected = pred_list[0]
        pred_list = np.append(pred_list, selected)
        pred_list = np.delete(pred_list, 0)


    #     x_zero = x_train[0]
    #     x_train = np.delete(x_train, 0, axis=0)
    #     x_train = np.append(x_train, x_zero)
    #     print("xtrain final?")
    #     print(x_train)

    out_file3 = 'output_generation.mid'
    out_pm = notes_to_midi(generated_song, out_file=out_file3)

    print("generated song")
    print(generated_song)

    return out_pm



