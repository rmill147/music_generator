import random
import numpy as np
from matplotlib import pylab
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from keras import Sequential
from keras.src.layers import SimpleRNN, Dense
from numpy.matlib import repmat
import tensorflow as tf
import csv
from tensorflow import keras
import pretty_midi

# dataset parameters ..... some of these need to be set/get methods
copies = 200       # number of times to copy the song
window_length = 12  # number of notes used to predict the next note
# neural network parameters
nn_nodes = 10      # number of nodes in the RNN
# training parameters
epochs = 600       # number of epochs used for training (cuts off early based on loss threshold + delta)
batch_size = None   # size of each batch (None is default)
song_length = 10
loss_threshold = 0.001

# TO-DO: Generate random notes to a song and see if the RNN will memorize
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
        if(logs.get('loss') is not None and logs.get('loss') <= loss_threshold):
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


def dataset_from_song(song,copies,window_length):
    # repeat song "copies" times
    songs = repmat(song,1,copies)[0] # number of notes, 1 row, each row repeats number of copies
    # number of windows used ; used to determine what gets added to training set... manipulate to specify number of notes
    # to predict?... (8,1,3) - 12
    num_windows = len(songs) - window_length
    #num_windows = song_length

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
sample_file = 'samples/suite_1_prelude_Gmaj_bach.mid'
midi_file = midi_to_notes(sample_file)

# song going in
song = midi_file

# print("Input song:")
# print(song)
# print("Input song length:")
# print(len(song))

# generate a dataset from copies of the song
#x_train,y_train = dataset_from_song(song,copies,window_length)

# RM: write out input song
# out_file = 'input.mid'
# in_pm = notes_to_midi(song, out_file=out_file)


def make_and_train(x_train,y_train):

    # specify the architecture of the neural network
    model = Sequential()
    model.add(SimpleRNN(nn_nodes,activation='relu'))
    model.add(Dense(1, activation=None))

    # set up the neural network
    model.compile(
        loss='MeanSquaredError',
        optimizer='Adam',
        metrics=[])
    # use this to save the best weights found so far
    # AC: can we set this so that the loss has to be less than 0.001
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


#model = make_and_train()
# shift the scale up by 2 half steps
test_song = song
print("Test song:")
print(test_song)

print("length of test song:")
print(len(test_song))
#copies_test = round((float(song_length) + 12.0)/float(len(test_song)))
copies_test = 3

# reverse the scale
# test_song = song[-1::-1]
# RM copies has effect on output; each time you add a copy
def do_test(model,test_song):
    x_test,y_test = dataset_from_song(test_song,copies_test,window_length)

    predictions = model.predict(x_test).round()

    # RM: Convert predictions to normal array for PM
    pred_list = predictions.reshape(-1)
    pred_list = pred_list.astype(int)

    correct = sum(predictions == y_test)
    N = len(predictions)
    test_accuracy = (correct/N)
    print(f"\nTest set accuracy: {100*correct/N}! ({correct}/{N})")
    #print(" test set accuracy: %.4f%% (%d/%d)" % (100*correct/N,correct,N))

    return pred_list, test_accuracy


#pred_list, test_accuracy = do_test(model)

# RM: write out output song

# out_file2 = 'output.mid'
# out_pm = notes_to_midi(pred_list, out_file=out_file2)
#
# print("Output song: ")
# print(pred_list)

# # 1: Plot Notes:nodes-> (x-axis)/accuracy -> (y-axis) -> for the memorization problem... maybe write out to csv file
# def plot(notes,nodes,accuracy,color="red"):
#     x = notes/nodes
#     y = accuracy
#
#     plt.scatter(x,y,color=color)
#     pylab.xlabel('notes/nodes')
#     pylab.ylabel('accuracy')
#     plt.show()
#
# ratio = len(song)/nn_nodes
# header = ['Notes','Nodes','Ratio','Accuracy']
# data = [len(song), nn_nodes, ratio, float(test_accuracy)]
# print(data)
#
#
# # open the file in the write mode
# f = open('data.csv', 'a', newline='')
# writer = csv.writer(f)
#
# # write a row to the csv file
# writer.writerow(data)
#
# # close the file
# f.close()

# plot data
#plot(len(song),nn_nodes,test_accuracy)

# 2. TO-DO: Generate a song ... can it learn a pattern to make a song?
# def predict_notes(song_length,x_train,song):
#     inputs = x_train
#
#     predictions = model.predict(inputs).round()
#
#     pitch = predictions[0]
#
#     return int(pitch)
#
# generated_song = [None] * len(pred_list)
# for i in range(0, len(pred_list)):
#     generated_song[i] = pred_list[i]
#
# for _ in range(song_length):
#     note = predict_notes(song_length,x_train, pred_list)
#     generated_song.append(note)
#     x_train = np.delete(x_train,0,axis=0)
#
#
# out_file3 = 'output_generation.mid'
# out_pm = notes_to_midi(generated_song, out_file=out_file3)
#
# print("generated song")
# print(generated_song)



