from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np

def seq_to_examples(img_captions, num_words=10000, seq_maxlen = 10):
    """
    `img_captions` is a list of caption data from COCOAPI, with format:
        {
          'image_id': 377421,
           'id': 56539,
           'caption': 'Four horses stand in outside, metal, holding pens.',
           'categories': [8, 19]
        }
    `num_words` is the max words in the vocabulary. Anything else replaced with 'UNK'
    `seq_maxlen` is the max number of words in the sequence. 
                 Each vector uses at most `seq_maxlen` - 1 words to predict the last word.
    """
    
    captions = []
    categories = []
    for c in img_captions:
        captions.append(c['caption'])
        categories.append(c['categories'])
    tokenizer = Tokenizer(num_words=num_words, oov_token='UNK')
    tokenizer.fit_on_texts(captions)
    
    encoded = tokenizer.texts_to_sequences(captions)
    
    # The partial sequences from the sentence
    # For example:
    #  'A man walking a dog' would turn into:
    #  ('A man walking a', 'dog'), ('A man walking', 'a'), ('A man', 'walking'), ('A', 'man')
    seqs = []
    # The word we are trying to predict
    preds = []
    # The new categories, in one-hot format
    y_categories = []
    for e, c in zip(encoded, categories):
        # Reindex the categories from 0 to 89 instead of 1 to 90
        # This is the format to_categorical expects
        c = np.array(c) - 1
        # Sum each of the one-hot vectors into a joined vector
        new_cat = to_categorical(c, num_classes=90).sum(axis=0)
        for i in range(1,len(e)):
            end_index = len(e) - i
            # Force the sequence to fit into seq_maxlen
            start_index = end_index - seq_maxlen
            if start_index < 0:
                start_index = 0            
            seqs.append(e[start_index:end_index])
            preds.append(e[-i])
            # Just add the same category over and over
            y_categories.append(new_cat)
    y_categories = np.stack(y_categories)

    # Make all sequences same length by adding 0 to end
    X = pad_sequences(seqs, padding='post')
    
    return X, y_categories