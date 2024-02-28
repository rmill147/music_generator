import numpy as np
import main
# import read_data
import csv
from main import midi_to_notes,dataset_from_song,make_and_train,shorten_song,do_test
# TODO: in main in addition to deletion of first in song generation, need to append at the end

v = main.Generator()

# Variables
copies = 200
window_length = 12  # number of notes used to predict the next note
# neural network parameters
node_num = int(input("Enter number of nodes: "))  # number of nodes in the RNN
# training parameters
num_rounds = 10
epoch_num = int(input("Enter number of epochs: "))



# ENSURE VARS ARE PASSED APPROPRIATELY
print(f"epochs: {epoch_num}")
print(f"nodes: {node_num}")

epoch_s = str(epoch_num)
nodes_s = str(node_num)

# Set up CSV
file_name = "test_epochs=" + epoch_s + "_nodes=" + nodes_s +".csv"
header = ['Notes', 'Nodes', 'Ratio', 'Accuracy', 'Song Title', 'Generated Song']
# open the file in the write mode
f = open(file_name, 'a', newline='')
writer = csv.writer(f)
# write a row to the csv file
writer.writerow(header)
# close the file
f.close()

# selecting a song... add desired songs here
sample_file = ['samples/cs1-1pre.mid', 'samples/cs2-4sar.mid', 'samples/cs2-3cou.mid',
               'samples/cs2-4sar.mid', 'samples/cs2-2all.mid', 'samples/cs2-1pre.mid', 'samples/cs1-6gig.mid',
              'samples/cs1-5men.mid', 'samples/cs1-4sar.mid', 'samples/cs1-3cou.mid', 'samples/cs1-2all.mid']

# sample_file = ['samples/cs2-4sar.mid']

# Iterate over each file
for x in range(len(sample_file)):

    print(f"song {x}: {sample_file[x]}")
    midi_file = midi_to_notes(sample_file[x])

    length = 100
    # song going in
    song = shorten_song(midi_file, length)

    for i in range(num_rounds):
        print(f"run {i} for song {x}!")
        # run main learning
        x_train, y_train = dataset_from_song(song, copies, window_length)
        model = make_and_train(x_train, y_train,node_num,epoch_num)
        test_song = [i+3 for i in song]
        pred_list, test_accuracy = do_test(model, test_song, window_length, length)

        # out_file2 = "output.mid" + str(i)
        # out_pm = main.notes_to_midi(pred_list, out_file=out_file2)

        print("Output song: ")
        print(pred_list)

        # accuracy with no shift in test?
        # run again with more nodes... 50 nodes
        # color code scatter plot
        ratio = len(song) / node_num
        data = [len(song), node_num, ratio, float(test_accuracy), sample_file[x], pred_list]
        print("data:")
        print(data)
        # open the file in the write mode
        f = open(file_name, 'a', newline='')
        writer = csv.writer(f)

        # write a row to the csv file
        writer.writerow(data)

        # close the file
        f.close()

        # Shorten song length
        length = length - 10
        song = shorten_song(midi_file, length)
