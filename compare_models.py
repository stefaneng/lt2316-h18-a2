import mycoco
import cocomodels
import utils
import pickle
import json

mycoco.setmode('train')

window_size = 5
# Memory error, changing to 8000
vocab_size = 8000
epochs = 20
batch_size = 256

# Get all the captions and categories
alliter = mycoco.iter_captions_cats()
allcaptions = list(alliter)

checkpointdir = "/scratch/gussteen/"

# Create the training data
X, y_words, y_categories, tokenizer = utils.seq_to_examples(allcaptions, num_words=vocab_size, seq_maxlen=window_size)
# print("Created {} training examples with window_size {}".format(X.shape[0], window_size))
# model, history = cocomodels.lstm_simple(X, y_words, y_categories, checkpointdir, vocab_size=vocab_size, batch_size=batch_size, epochs = epochs, logfile="compare_simple.csv")

# print(history.history)
# model.save('/scratch/gussteen/lstm_simple.hdf5')
# with open('./history_lstm_simple.json', 'w+') as f:
#    json.dump(history.history, f)

# 0.1 dropout
model, history = cocomodels.lstm_complex(X, y_words, y_categories, checkpointdir, vocab_size = vocab_size, batch_size=batch_size, epochs = epochs, dropout=0.1, logfile="compare_dropout1.csv")
print(history.history)
model.save('/scratch/gussteen/lstm_complex_5_window_1_drop.hdf5')
with open('./history_lstm_complex_5_window_1_drop.json', 'w+') as f:
    json.dump(history, f)

# 0.5 dropout
model, history = cocomodels.lstm_complex(X, y_words, y_categories, checkpointdir, vocab_size = vocab_size, batch_size=batch_size, epochs=epochs, dropout=0.1, logfile="compare_dropout5.csv")
print(history.history)
model.save('/scratch/gussteen/lstm_complex_5_window_5_drop.hdf5')
with open('./history_lstm_complex_5_window_5_drop.json', 'w+') as f:
    json.dump(history.history, f)

window_size = 10
del X
del y_words
del y_categories
X, y_words, y_categories, tokenizer = utils.seq_to_examples(allcaptions, num_words=vocab_size, seq_maxlen=window_size)
# 0.1 dropout with larger window size
model, history = cocomodels.lstm_complex(X, y_words, y_categories, checkpointdir, vocab_size=vocab_size, batch_size=batch_size, epochs=epochs, dropout=0.1, logfile="compare_window10.csv")
print(history.history)
model.save('/scratch/gussteen/lstm_complex_10_window_1_drop.hdf5')
with open('./history_lstm_complex_10_window_1_drop', 'w+') as f:
    json.dump(history.history, f)  
