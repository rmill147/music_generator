import numpy as np
import main
# import read_data
import csv

# TODO: in main in addition to deletion of first in song generation, need to append at the end


# Variables
copies = 200  # number of times to copy the song
window_length = 12  # number of notes used to predict the next note
# neural network parameters
nn_nodes = 10  # number of nodes in the RNN
# training parameters
num_rounds = 1

# selecting a song... add desired songs here
# sample_file = ['samples/cs1-1pre.mid', 'samples/cs2-4sar.mid', 'samples/cs2-3cou.mid',
               #'samples/cs2-4sar.mid', 'samples/cs2-2all.mid', 'samples/cs2-1pre.mid', 'samples/cs1-6gig.mid',
              # 'samples/cs1-5men.mid', 'samples/cs1-4sar.mid', 'samples/cs1-3cou.mid', 'samples/cs1-2all.mid']

sample_file = ['samples/cs1-1pre.mid']

# Iterate over each file
for x in range(len(sample_file)):
    print(f"song {x}: {sample_file[x]}")
    midi_file = main.midi_to_notes(sample_file[x])

    length = 100
    # song going in
    song = main.shorten_song(midi_file, length)

    for i in range(num_rounds):
        print(f"run {i} for song {x}!")
        # run main learning
        x_train, y_train = main.dataset_from_song(song, copies, window_length)
        model = main.make_and_train(x_train, y_train)
        test_song = [i+3 for i in song]
        pred_list, test_accuracy = main.do_test(model, test_song)

        # out_file2 = "output.mid" + str(i)
        # out_pm = main.notes_to_midi(pred_list, out_file=out_file2)

        print("Output song: ")
        print(pred_list)

        # accuracy with no shift in test?
        # run again with more nodes... 50 nodes
        # color code scatter plot
        ratio = len(song) / nn_nodes
        header = ['Notes', 'Nodes', 'Ratio', 'Accuracy', 'song']
        data = [len(song), nn_nodes, ratio, float(test_accuracy), sample_file[x]]
        print("data:")
        print(data)

        # open the file in the write mode
        f = open('run_3_10_nodes_600_epochs.csv', 'a', newline='')
        writer = csv.writer(f)

        # write a row to the csv file
        writer.writerow(data)

        # close the file
        f.close()

        # Shorten song length
        length = length - 10
        song = main.shorten_song(midi_file, length)
