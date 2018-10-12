from keras.models import Model
from keras.layers import Dense, LSTM, Embedding, Input, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import CSVLogger

earlystopping = EarlyStopping(monitor='loss', patience=2)

def lstm_simple(X, y_words, y_categories, checkpointdir, vocab_size = 10000, batch_size = 256, epochs = 20, embed_size = 50, logfile = 'model_log.csv', load_from=None):
    "y = [y_words, y_categories]"
    input_length = X.shape[1]
    cats_length = y_categories.shape[1]

    ## Model 1
    inputs = Input(shape=(input_length,))
    embed = Embedding(vocab_size, embed_size, input_length=input_length)(inputs)
    lstm = LSTM(50, dropout=0.1)(embed)
    # Word prediction softmax
    word_pred = Dense(vocab_size, activation='softmax', name='word_prediction')(lstm)
    # 90 categories, sigmoid activation
    category_preds = Dense(cats_length, activation = 'sigmoid', name='category_prediction')(lstm)

    # This creates a model that includes
    # the Input layer and two Dense layers outputs
    model = Model(inputs=inputs, outputs=[word_pred, category_preds])

    model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    
    if load_from:
        print("Loading weights from:", load_from)
        model.load_weights(load_from)

    # Checkpointing and logging
    csv_logger = CSVLogger(logfile, append=True, separator=',')
    filepath= checkpointdir + "lstm_simple.{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, verbose=1)
    
    history = model.fit(X, [y_words, y_categories], batch_size=batch_size, callbacks=[earlystopping, checkpoint, csv_logger], epochs=epochs)
    return model, history

def lstm_complex(X, y_words, y_categories, checkpointdir, vocab_size = 10000, batch_size = 256, epochs = 20, embed_size = 50, dropout=0.1, logfile = 'model_log.csv', load_from=None):
    "y = [y_words, y_categories]"
    input_length = X.shape[1]
    cats_length = y_categories.shape[1]

    ## Model 1
    inputs = Input(shape=(input_length,))
    embed = Embedding(vocab_size, embed_size, input_length=input_length)(inputs)
    dropout_layer = Dropout(dropout)(embed)
    # Two levels of LSTM, each with dropout
    lstm1 = LSTM(50, return_sequences=True, dropout=dropout)(dropout_layer)
    lstm2 = LSTM(50, dropout=dropout)(lstm1)
    # Word prediction softmax
    word_pred = Dense(vocab_size, activation='softmax', name='word_prediction')(lstm2)
    # cats_length categories, sigmoid activation
    category_preds = Dense(cats_length, activation = 'sigmoid', name='category_prediction')(lstm2)

    # This creates a model that includes
    # the Input layer and two Dense layers outputs
    model = Model(inputs=inputs, outputs=[word_pred, category_preds])

    model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    if load_from:
        print("Loading weights from:", load_from)
        model.load_weights(load_from)

    # Checkpointing and logging
    csv_logger = CSVLogger(logfile, append=True, separator=',')
    filename = "lstmdouble_dropout{}_window{}".format(dropout, input_length)
    filepath= checkpointdir + filename + ".{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, verbose=1)
    
    history = model.fit(X, [y_words, y_categories], batch_size=batch_size, callbacks=[earlystopping, checkpoint, csv_logger], epochs=epochs)
    return model, history
