# This is the main testing script that we should be able to run to grade
# your model training for the assignment.
# You can create whatever additional modules and helper scripts you need,
# as long as all the training functionality can be reached from this script.

# Add/update whatever imports you need.
import pickle
import json
from argparse import ArgumentParser
from keras.models import load_model

import utils
import mycoco

# If you do option A, you may want to place your code here.  You can
# update the arguments as you need.
def optA():
    mycoco.setmode('test')
    print("Option A not implemented!")

# If you do option B, you may want to place your code here.  You can
# update the arguments as you need.
def optB(modelfile, traintokenizer, maxinstances):
    with open(traintokenizer, 'rb') as f:
        tokenizer = pickle.load(f)
    with open('./categories_idindex.json') as f:
        cat_dict = json.load(f)
        
    vocab_size = tokenizer.num_words
    
    mycoco.setmode('test')

    model = load_model(modelfile)
    
    # Get the window size from the model input
    window_size = model.layers[0].get_input_at(0).get_shape().as_list()[1]

    alliter = mycoco.iter_captions_cats(maxinstances=maxinstances)
    allcaptions = list(alliter)       

    model.summary()

    print("Found:", len(list(allcaptions)), "captions in test set")


    X, y_words, y_categories, tokenizer2 = utils.seq_to_examples(allcaptions, cat_dict, num_words=vocab_size, seq_maxlen=window_size, tokenizer=tokenizer)

    print("Created {} test examples with window size {}".format(X.shape[0], window_size))

    score = model.evaluate(X, [y_words, y_categories])
    # Not sure why category_prediction_loss is 0?
    for m, s in zip(model.metrics_names, score):
        print("Metric {}: {}".format(m, s))

# Modify this as needed.
if __name__ == "__main__":
    parser = ArgumentParser("Evaluate a model.")
    # Add your own options as flags HERE as necessary (and some will be necessary!).
    # You shouldn't touch the arguments below.
    parser.add_argument('-P', '--option', type=str,
                        help="Either A or B, based on the version of the assignment you want to run. (REQUIRED)",
                        required=True)
    parser.add_argument('-m', '--maxinstances', type=int,
                        help="The maximum number of instances to be processed per category. (optional)",
                        required=False)
    parser.add_argument('modelfile', type=str, help="model file to evaluate")
    parser.add_argument('tokenizer', type=str, help="Saved tokenizer file from training run. (REQUIRED)")
    args = parser.parse_args()

    print("Output model in " + args.modelfile)
    print("Maximum instances is " + str(args.maxinstances))

    print("Executing option " + args.option)
    if args.option == 'A':
        optA()
    elif args.option == 'B':
        optB(args.modelfile, args.tokenizer, args.maxinstances)
    else:
        print("Option does not exist.")
        exit(0)
