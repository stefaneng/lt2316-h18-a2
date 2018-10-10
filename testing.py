import numpy as np
import random
from numpy import array
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Embedding, Input, concatenate
from keras.callbacks import ModelCheckpoint

import mycoco
import utils

mycoco.setmode('train')

alliter = mycoco.iter_captions_cats()
allcaptions = list(alliter)

vocab_size=10000

X, y_words, y_categories, tokenizer = utils.seq_to_examples(allcaptions, num_words=vocab_size)

# print(X.shape, y_categories.shape)

# Randomly sample for testing purposes
sample = np.random.choice(X.shape[0], 10000, replace=False)

input_length = X.shape[1]
inputs = Input(shape=(input_length,))
embed = Embedding(vocab_size, 50, input_length=input_length)(inputs)
lstm = LSTM(50, dropout=0.1)(embed)
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

# Checkpointing
filepath="/scratch/gussteen/testing_weights.{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, verbose=1)

model.fit(X, [y_words, y_categories], batch_size=128, callbacks=[checkpoint], epochs=5)
