import pickle
import json
from argparse import ArgumentParser
from keras.models import load_model
from keras.metrics import categorical_accuracy
from keras.losses import categorical_crossentropy
import keras.backend as K

import utils
import mycoco

def perplexity_metric(y_true, y_pred):
    cross_entropy = categorical_crossentropy(y_true, y_pred)
    return K.pow(2.0, cross_entropy)

def perplexity(modelfile, traintokenizer, maxinstances):
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
    
    model.compile(optimizer='adam',
        loss={
            'word_prediction': 'categorical_crossentropy',
            'category_prediction': 'binary_crossentropy'
        },
        metrics=[
            perplexity_metric
        ])

    print("Found:", len(list(allcaptions)), "captions in test set")


    X, y_words, y_categories, tokenizer2 = utils.seq_to_examples(allcaptions, cat_dict, num_words=vocab_size, seq_maxlen=window_size, tokenizer=tokenizer)

    print("Created {} test examples with window size {}".format(X.shape[0], window_size))

    score = model.evaluate(X, [y_words, y_categories])
    
    metrics = dict(zip(model.metrics_names, score))
    print("Perplexity 2^(word cross entropy) =", metrics['word_prediction_perplexity_metric'])

if __name__ == "__main__":
    parser = ArgumentParser("Compute the perplexity of a model.")
    parser.add_argument('-m', '--maxinstances', type=int,
                        help="The maximum number of instances to be processed per category. (optional)",
                        required=False)
    parser.add_argument('--windowsize', type=int,
                    help="The window size used in training (REQUIRED)")    
    parser.add_argument('modelfile', type=str, help="model file to evaluate")
    parser.add_argument('tokenizer', type=str, help="Saved tokenizer file from training run. (REQUIRED)")
    args = parser.parse_args()

    perplexity(args.modelfile, args.tokenizer, args.maxinstances)
