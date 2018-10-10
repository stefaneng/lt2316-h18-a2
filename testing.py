import numpy as np
import random
from numpy import array
from keras.utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Embedding, Input, concatenate
from keras.callbacks import ModelCheckpoint

import pickle

# Here is where we would replace the sample with the full dataset
with open('./horse_dog_sample.pickle', 'rb') as f:
    sample = pickle.load(f)

# Take a even smaller sample to test checkpoint, etc.
sample = random.sample(sample, k=1000)

# Separate out the captions and categories
captions = []
categories = []
for c in sample:
    captions.append(c['caption'])
    categories.append(c['categories'])

# May need to set some max length here
# For now we just let the tokenizer grow to
tokenizer = Tokenizer()
tokenizer.fit_on_texts(["\n".join(captions)])
encoded = tokenizer.texts_to_sequences(captions)
encoded

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
        seqs.append(e[:end_index])
        preds.append(e[-i])
        # Just add the same category over and over
        y_categories.append(new_cat)
y_categories = np.stack(y_categories)

# Make all sequences same length by adding 0 to end
X = pad_sequences(seqs, padding='post')
input_length = len(X[0])
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
y_words = to_categorical(preds, num_classes=vocab_size)

inputs = Input(shape=(input_length,))
embed = Embedding(vocab_size, 10, input_length=input_length)(inputs)
lstm = LSTM(10)(embed)
# Word prediction softmax
word_pred = Dense(vocab_size, activation='softmax', name='word_prediction')(lstm)
# 90 categories, sigmoid activation
category_preds = Dense(90, activation = "sigmoid", name='category_prediction')(lstm)

# This creates a model that includes
# the Input layer and two Dense layers outputs
model = Model(inputs=inputs, outputs=[word_pred, category_preds])

# Load from a checkpoint if we need to
# model.load_weights("weights.{epoch:02d}.hdf5")

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())
plot_model(model, to_file='testing_model.png')

# Checkpointing
filepath="weights.{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, verbose=1)

model.fit(X, [y_words, y_categories], callbacks=[checkpoint], epochs=5)

# TODO: Prediction
