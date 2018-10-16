# This is the main testing script that we should be able to run to grade
# your model training for the assignment.
# You can create whatever additional modules and helper scripts you need,
# as long as all the training functionality can be reached from this script.

# Add/update whatever imports you need.
import pickle
import numpy as np
from argparse import ArgumentParser
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

import json
import utils
import mycoco

# If you do option A, you may want to place your code here.  You can
# update the arguments as you need.
def optA():
    mycoco.setmode('test')
    print("Option A not implemented!")

# If you do option B, you may want to place your code here.  You can
# update the arguments as you need.
def predict(predict_sent, modelfile, traintokenizer, window_size, npredictions):
    with open(traintokenizer, 'rb') as f:
        tokenizer = pickle.load(f)
        
    with open('./categories_idindex.json') as f:
        cat_dict = json.load(f)

    vocab_size = tokenizer.num_words    
    
    model = load_model(modelfile)
    model.summary()
    
    encoded = tokenizer.texts_to_sequences([predict_sent])
    # Flip the word index around so we can look up word names based on the index
    word_lookup = {v: k for k, v in tokenizer.word_index.items()}

    predicted_words = []
    for i in range(npredictions):
        x = pad_sequences(encoded, padding='post', truncating='pre', maxlen=window_size)

        word_preds, cat_preds = model.predict(np.array(x))
        
        # Only are predicting one value
        words_preds = word_preds[0]
        cat_preds = cat_preds[0]

        # Use this prediction as the last word
        pred = np.argmax(word_preds, axis=None) + 1
        predicted_words.append(word_lookup[pred])
        encoded[0].append(pred)
    
    if npredictions == 1:
        # Word predictions
        # Descending order
        sort_word_preds = np.argsort(word_preds, axis=None)[::-1]
        sort_word_names = [word_lookup[i + 1] for i in sort_word_preds]
        sort_word_probs = words_preds[sort_word_preds]

        print("Predicting: {}...".format(predict_sent))
        print("Word Predictions:")
        for w, prob in list(zip(sort_word_names, sort_word_probs))[:5]:
            print("{}: {}".format(w, prob))
    
        print("---------")
    elif npredictions > 1:
        print("Predicting {} words: {} -> {}".format(npredictions, predict_sent, " ".join(predicted_words)))
    else:
        print("Number of predictions must be >= 1")
        return
    
    # Category
    # Create a mapping with key=index and value=name so we can easily get back the category names
    cats_id_name = {int(k['index']): k['name'] for k in cat_dict.values()}
    # Get the ordered indices
    sort_cat_preds = np.argsort(cat_preds, axis=None)[::-1]
    # Get the names of the top 
    sort_cat_names = [cats_id_name[s] for s in sort_cat_preds]
    # Get top 5 predictions with probabilities
    sort_cat_probs = cat_preds[sort_cat_preds]
    sort_cat_probs /= sum(sort_cat_probs)
 
    print("Category Predictions:")
    # Only take the top 5 categories
    for c,prob in list(zip(sort_cat_names, sort_cat_probs))[:5]:
        print("{}: {}".format(c, prob))


# Modify this as needed.
if __name__ == "__main__":
    parser = ArgumentParser("Predict the next N words in a sentence.")
    parser.add_argument('sentence', type=str, help="Sentence we are trying to predict last word")
    parser.add_argument('modelfile', type=str, help="model file to evaluate")
    parser.add_argument('tokenizer', type=str, help="Saved tokenizer file from training run. (REQUIRED)")
    parser.add_argument('--windowsize', type=int, help="Size of window. Must be the size given in `modelfile` (REQUIRED)")
    parser.add_argument('-n', '--npredictions', type=int, help="Number of predictions to make (Default 1)", nargs='?', const=1)
    args = parser.parse_args()

    predict(args.sentence, args.modelfile, args.tokenizer, args.windowsize, args.npredictions)
