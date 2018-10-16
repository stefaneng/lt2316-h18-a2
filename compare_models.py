import mycoco
import cocomodels
import utils
import pickle
import json

mycoco.setmode('train')

window_size = 3
vocab_size = 10000
epochs = 10
batch_size = 256
embed_size = 50

# maxinstances = 250
maxinstances = None

# Get all the captions and categories
alliter = mycoco.iter_captions_cats(maxinstances=maxinstances)
allcaptions = list(alliter)

checkpointdir = "/scratch/gussteen/"

with open('./categories_idindex.json') as f:
    cat_dict = json.load(f)

# Create the training data
X, y_words, y_categories, tokenizer = utils.seq_to_examples(allcaptions, cat_dict, num_words=vocab_size, seq_maxlen=window_size)

# Save the tokenizer to use for testing

maxinstance_str = "_maxinstance" + str(maxinstances) if maxinstances else ""
with open('./tokenizer{}{}.pickle'.format(vocab_size, maxinstance_str), 'wb') as f:
    pickle.dump(tokenizer, f)

print("Created {} training examples with window_size {}".format(X.shape[0], window_size))
model, history = cocomodels.lstm_simple(X, y_words, y_categories, checkpointdir, vocab_size=vocab_size, batch_size=batch_size, epochs = epochs, embed_size = embed_size, logfile="./results/compare_simple_2.csv")

print(history.history)
model.save('./models/lstm_simple.hdf5')
with open('./results/history_lstm_simple.json', 'w+') as f:
    json.dump(history.history, f)
    
# Increase word embedding size, use simple model
embed_size = 100
# 0.1 dropout with larger window size
model, history = cocomodels.lstm_simple(X, y_words, y_categories, checkpointdir, vocab_size=vocab_size, batch_size=batch_size, epochs=epochs, embed_size = embed_size, logfile="./results/compare_embed_100_2.csv")
print(history.history)
model.save('./models/lstm_simple_embed100_2.hdf5')
with open('./results/history_lstm_simple_embed100_2.json', 'w+') as f:
    json.dump(history.history, f)    

# 0.1 dropout
model, history = cocomodels.lstm_complex(X, y_words, y_categories, checkpointdir, vocab_size = vocab_size, batch_size=batch_size, epochs = epochs, dropout=0.1, embed_size = embed_size, logfile="./results/compare_complex_embed_100_do1_2.csv")
print(history.history)
model.save('./models/lstm_complex_drop1_2.hdf5')
with open('./results/history_lstm_complex_drop1_2.json', 'w+') as f:
    json.dump(history.history, f)

# 0.5 dropout
model, history = cocomodels.lstm_complex(X, y_words, y_categories, checkpointdir, vocab_size = vocab_size, batch_size=batch_size, epochs=epochs, dropout=0.5, embed_size = embed_size, logfile="./results/compare_complex_embed_100_do5_2.csv")
print(history.history)
model.save('./moels/lstm_complex_drop5_2.hdf5')
with open('./results/history_lstm_complex_drop5_2.json', 'w+') as f:
    json.dump(history.history, f)
