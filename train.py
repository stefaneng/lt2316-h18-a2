# This is the main training script that we should be able to run to grade
# your model training for the assignment.
# You can create whatever additional modules and helper scripts you need,
# as long as all the training functionality can be reached from this script.

# Add/update whatever imports you need.
from argparse import ArgumentParser
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

import mycoco
import cocomodels
import utils
import json

# If you do option A, you may want to place your code here.  You can
# update the arguments as you need.
def optA():
    mycoco.setmode('train')
    print("Option A not implemented!")

# If you do option B, you may want to place your code here.  You can
# update the arguments as you need.
def optB(init_model, categories, out_model, maxinstances, window_size, checkpointdir):
    # TODO: Other values to add as parameters
    # Number of previous words to use in prediction
    vocab_size = 8000
    epochs = 3
    batch_size = 256
    logfile = checkpointdir + "train_results.csv"
    mycoco.setmode('train')

    if init_model:
        # Load the model and only train on the given categories
        model = load_model(init_model)
    else:
        # Re-train the model on the entired caption dataset

        # Get all the captions and categories
        alliter = mycoco.iter_captions_cats(maxinstances=maxinstances)
        allcaptions = list(alliter)

        with open('./categories_idindex.json') as f:
            cat_dict = json.load(f)

        # Create the training data
        X, y_words, y_categories, tokenizer = utils.seq_to_examples(allcaptions, cat_dict, num_words=vocab_size, seq_maxlen=window_size)
        print("Created {} training examples with window_size {}".format(X.shape[0], window_size))
        model, history = cocomodels.lstm_simple(X, y_words, y_categories, checkpointdir,
                            vocab_size=vocab_size, batch_size = batch_size, epochs = epochs, logfile = logfile)
        with open(checkpointdir + out_model + 'history.json', 'w+') as fh:
            print("Saved history to", checkpointdir + out_model + 'history')
            print(history.history)
            json.dump(history.history, fh)

    # Captions for the given categories
    alliter = mycoco.iter_captions_cats(categories, maxinstances=maxinstances)
    allcaptions = list(alliter)

    # Re-train on just the given categories
    X, y_words, y_categories, tokenizer = utils.seq_to_examples(allcaptions, cat_dict, num_words=vocab_size, seq_maxlen=window_size)
    print("Created {} training examples for categories {} with window_size {}".format(X.shape[0], ",".join(categories), window_size))

    cat_joined = "_".join(categories)
    csv_logger = CSVLogger(logfile, append=True, separator=';')
    filepath= checkpointdir + "lstm_simple" + cat_joined + "epoch{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, verbose=1)

    history = model.fit(X, [y_words, y_categories], batch_size=batch_size, callbacks=[checkpoint, csv_logger], epochs=epochs)
    with open(checkpointdir + out_model + cat_joined + '_history.json', 'w+') as fh:
        print("Saved history to", checkpointdir + out_model + 'history')
        print(history.history)
        json.dump(history.history, fh)
    model.save(out_model)

# Modify this as needed.
if __name__ == "__main__":
    parser = ArgumentParser("Train a model.")
    # Add your own options as flags HERE as necessary (and some will be necessary!).
    parser.add_argument('--init_model', type=str, help="starting model. Will skip training on entire data set if provided and only retrain on the given categories. (optional)", required=False)
    # You shouldn't touch the arguments below.
    parser.add_argument('-P', '--option', type=str,
                        help="Either A or B, based on the version of the assignment you want to run. (REQUIRED)",
                        required=True)
    parser.add_argument('-m', '--maxinstances', type=int,
                        help="The maximum number of instances to be processed per category. (optional)",
                        required=False)
    parser.add_argument('--windowsize', type=int,
                        help="The window size (optional)",
                        nargs='?', const=3, default=3)
    parser.add_argument('checkpointdir', type=str,
                        help="directory for storing checkpointed models and other metadata (recommended to create a directory under /scratch/)")
    parser.add_argument('modelfile', type=str, help="output model file")
    parser.add_argument('categories', metavar='cat', type=str, nargs='+',
                        help='two or more COCO category labels')
    args = parser.parse_args()

    print("Output model in " + args.modelfile)
    print("Working directory at " + args.checkpointdir)
    print("Maximum instances is " + str(args.maxinstances))
    print("Window size is " + str(args.windowsize))

    if len(args.categories) < 2:
        print("Too few categories (<2).")
        exit(0)

    print("The queried COCO categories are:")
    for c in args.categories:
        print("\t" + c)

    print("Executing option " + args.option)
    if args.option == 'A':
        optA()
    elif args.option == 'B':
        optB(args.init_model, args.categories, args.modelfile, args.maxinstances, args.windowsize, args.checkpointdir)
    else:
        print("Option does not exist.")
        exit(0)
