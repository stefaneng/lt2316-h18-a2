# This is the main training script that we should be able to run to grade
# your model training for the assignment.
# You can create whatever additional modules and helper scripts you need,
# as long as all the training functionality can be reached from this script.

# Add/update whatever imports you need.
from argparse import ArgumentParser
import mycoco
import cocomodels
import utils

# If you do option A, you may want to place your code here.  You can
# update the arguments as you need.
def optA():
    mycoco.setmode('train')
    print("Option A not implemented!")

# If you do option B, you may want to place your code here.  You can
# update the arguments as you need.
def optB(init_model, categories, out_model, maxinstances, checkpointdir):
    # TODO: Other values to add as parameters
    # Number of previous words to use in prediction
    window_size = 5
    vocab_size = 100
    mycoco.setmode('train')

    if init_model:
        # Load the model and only train on the given categories
        model = load_model(init_model)
    else:
        # Re-train the model on the entired caption dataset

        # Get all the captions and categories
        alliter = mycoco.iter_captions_cats()
        allcaptions = list(alliter)

        # Create the training data
        X, y_words, y_categories, tokenizer = utils.seq_to_examples(allcaptions, num_words=vocab_size, seq_maxlen=window_size)
        print("Created {} training examples with window_size {}".format(X.shape[0], window_size))
        model, history = cocomodels.lstm_simple(X, y_words, y_categories, checkpointdir, vocab_size=vocab_size, batch_size = 256, epochs = 3)
        print(history.history)

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
    parser.add_argument('checkpointdir', type=str,
                        help="directory for storing checkpointed models and other metadata (recommended to create a directory under /scratch/)")
    parser.add_argument('modelfile', type=str, help="output model file")
    parser.add_argument('categories', metavar='cat', type=str, nargs='+',
                        help='two or more COCO category labels')
    args = parser.parse_args()

    print("Output model in " + args.modelfile)
    print("Working directory at " + args.checkpointdir)
    print("Maximum instances is " + str(args.maxinstances))

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
        optB(args.init_model, args.categories, args.modelfile, args.maxinstances, args.checkpointdir)
    else:
        print("Option does not exist.")
        exit(0)
