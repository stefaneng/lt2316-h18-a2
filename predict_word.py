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

import utils
import mycoco

# If you do option A, you may want to place your code here.  You can
# update the arguments as you need.
def optA():
    mycoco.setmode('test')
    print("Option A not implemented!")

# If you do option B, you may want to place your code here.  You can
# update the arguments as you need.
def predict(predict_sent, modelfile, traintokenizer):
    with open(traintokenizer, 'rb') as f:
        tokenizer = pickle.load(f)
        
    window_size = 5
    vocab_size = tokenizer.num_words    
    
    model = load_model(modelfile)
#    print(model.summary())
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    encoded = tokenizer.texts_to_sequences([predict_sent])
    # First remove all words so we have at most `window_size`
    # Then pad the remaining last as 0
    x = pad_sequences(encoded, padding='post', truncating='pre', maxlen=window_size)
    
    word_preds, cat_preds = model.predict(np.array(x))
    # Add 1 to get actual category since we subtracted one in one-hot encoding
    words_preds = word_preds[0]
    cat_preds = cat_preds[0]
    
    # Word predictions
    # sort_word_preds = np.argsort(word_preds)
    
    
    # Category
    cats = mycoco.get_categories()
    sort_cat_preds = np.argsort(cat_preds)
    sort_cat_names = [] # [cats[s] for s in sort_cat_preds] when I change categories to 80
    for s in sort_cat_preds:
        if s in cats:
            sort_cat_names.append(cats[s])
        else:
            # Workaround because I trained on 90 categories instead of 80...
            sort_cat_names.append(s)
    # Get top 5 predictions with probabilities
    sort_cat_probs = cat_preds[sort_cat_preds]
    sort_cat_probs /= sum(sort_cat_probs)
        
    print("Category Predictions:")
    for c,prob in zip(sort_cat_names, sort_cat_probs):
        print("{}: {}".format(c, prob))


# Modify this as needed.
if __name__ == "__main__":
    parser = ArgumentParser("Predict the next N words in a sentence.")
    parser.add_argument('sentence', type=str, help="Sentence we are trying to predict last word")
    parser.add_argument('modelfile', type=str, help="model file to evaluate")
    parser.add_argument('tokenizer', type=str, help="Saved tokenizer file from training run. (REQUIRED)")
    args = parser.parse_args()

    predict(args.sentence, args.modelfile, args.tokenizer)