# LT2316 H18 Assignment 2

## Usage

### Example

#### Train Model
Train the model on *horse* and *cat* categories.
```
python train.py -P B --maxinstances=25 ./test/ ./test/test_model.hdf5 'horse' 'cat'
```
  - Model will be saved in `./test/test_model.hdf5` and
  - Tokenizer saved to `./test/tokenizer10000_horse_cat.pickle`
Both of these need to be provided to `test.py`, but better to use the pretrained models and tokenizers. Just copy the examples below

#### Test Model
Then to test the model using the best model (not the one we just trained on)
```
python test.py -P B --maxinstances=50 ./models/lstm_simple_embed100.hdf5 ./tokenizer10000.pickle
```

#### Perplexity
Calculate the perplexity of the model
```
python perplexity.py --maxinstances=50 ./models/lstm_simple_embed100.hdf5 ./tokenizer10000.pickle
```

#### Predict Words
```
python predict_word.py "A man is sitting on a" ./models/lstm_simple_embed100.hdf5 ./tokenizer10000.pickle
```

    Predicting: A man is sitting on a...
    Word Predictions:
    picture: 0.11356481909751892
    holding: 0.11310586333274841
    window: 0.04701809585094452
    glass: 0.04402976483106613
    looking: 0.038740649819374084
    ---------
    Category Predictions:
    person: 0.14352022111415863
    chair: 0.045445144176483154
    cat: 0.04415004700422287
    dining table: 0.04063219204545021
    bench: 0.03427702933549881

Multi-word prediction
```
python predict_word.py -n 3 "A man is sitting on a" ./models/lstm_simple_embed100.hdf5 ./tokenizer10000.pickle
```

    Predicting 3 words: A man is sitting on a -> picture the man
    Category Predictions:
    person: 0.11248820275068283
    chair: 0.07324380427598953
    tv: 0.06896624714136124
    book: 0.04803682491183281
    couch: 0.04134979099035263

###  Model Diagram
```python
from keras.models import load_model
from keras.utils import plot_model

model = load_model('./models/lstm_simple_embed100.hdf5')
plot_model(model, to_file="./imgs/lstm_simple_embed100.jpg", show_shapes=True)
```

![Simple](imgs/lstm_simple_embed100.jpg)

### Architecture Overview

After experiementing with many different architectures and LSTM layers, what ended up working the best was a single LSTM layer with 50 nodes. Each word is represented as an integer less than 10000. Some of the other approaches I tried were to have the embedding layer input into two different LSTM layers which in turn seperately predicted either word prediction, or category prediction. This performed worse than other options as well. The other architectures I experimented with are described below, where two LSTM layers are used (passing sequences from first to the second).

I had originally started with a window size of 5 and then tested a window size of 10. The accuracy was a little better but when using word prediction script the results did not appear as good. The tradeoff of training time was also not worth it as the window size of 10 took almost twice as long to run.

  - **Loss function**: Categorical Cross Entropy for the word prediction since we have one-hot vector encodings and Binary Cross Entropy for the multi-category classification.
  - **Batch Size**: 256
  - **Epochs**: 10 (For final test)
  - **Window size**: 3
  - **Data size**: Using the window size of 10, I created 5572508 training examples from the 591753 captions in the training data. In the test set, 236352 training examples were created from 25014 captions.
  - **Dropout**: dropout was set to 0.1 for the LSTM layer. This performed better than no dropout and 0.5 dropout. It was also not the best test for this as I think with many more epochs it could have performed the same and generalized better.

### Variant
![Two LSTM Layers](./imgs/lstm_complex_drop1.jpg)

This architectural variant did not seem to perform as well at the simple one layer LSTM model. I experimented with setting the dropout rate which increased the loss. This might not have been fair to do, since I did not run it for many epochs. Increasing the dropout rate should make the model converge slower but overfit less often which I didn't give it the chance to do with such a small number of epochs. This is because each epoch still took a fair amount of time (~10 - 20 minutes) and I was experimenting with many different hyperparameters.

Code below
```python
from keras.models import load_model
from keras.utils import plot_model

# Alternative model
model = load_model('./models/lstm_complex_drop1.hdf5')
plot_model(model, to_file="./imgs/lstm_complex_drop1.jpg", show_shapes=True)
```

### Hyperparameter Selection  

  After experiementing with many different architectures and LSTM layers, what ended up working the best was a single LSTM layer with 50 nodes. I first changed the embedding size to 100 and then for the two other variations a more complex model (seen above) with additional dropout at levels 0.1 and 0.5. I thought that adding another LSTM layer would perform better than the single layer. It is possible that increasing the epochs or modifying the other hyperparameters would have made a difference. Also batch size was set to 256 as it allowed me to actually train my models in a reasonable amount of time. With a large number of examples, 256 seemed to be a good fit. The window size was experimented with a lot. I started with 5, and then went to 10. The loss decreased as I went to 10 but the training time went up substantially. When I actually tested the results on the predict word script it seemed to be more realistic with a window of size 3. It also allowed me to test out more hyperparameters in the time-alloted for this assignment.

### Model Loss for Different Hyperparameters
![Model Loss](./imgs/model_loss.png)

The final model I went with was the [*Simple LSTM 100 Embed*](models/lstm_simple_embed100.hdf5). This was an increase from the base model from 50 node embedding layer to 100. The more complex models (with an additional LSTM layer and dropout) did not perform as well overall.

Code to generate the plot is show below


```python
import matplotlib.pyplot as plt

import pandas as pd

model_simple = pd.read_csv('./results/compare_simple.csv')
model_simple_embed100 = pd.read_csv('./results/compare_embed_100.csv')
model_complex_drop1 = pd.read_csv('./results/compare_complex_embed_100_do1.csv')
model_complex_drop5 = pd.read_csv('./results/compare_complex_embed_100_do5.csv')

# Plot overall loss
plt.plot(model_simple['epoch'], model_simple['loss'], label='Simple LSTM 50 Embed')
plt.plot(model_simple_embed100['epoch'], model_simple_embed100['loss'], label='Simple LSTM 100 Embed')
plt.plot(model_complex_drop1['epoch'], model_complex_drop1['loss'], label='Complex LSTM 100 Embed Dropout 0.1')
plt.plot(model_complex_drop5['epoch'], model_complex_drop5['loss'], label='Complex LSTM 100 Embed Dropout 0.5')

plt.legend(loc='upper right')
plt.title("Overall Cross Entropy Loss")

plt.savefig('./imgs/model_loss.png', bbox_inches='tight')
```

### Lessons Learned

I learned lots about language models from building this word and category prediction. Also this project was a nice introduction to the Keras functional API. It was interesting to experiment with branching into two seperate LSTM models and combining at the end. The functional API is very simple to use and extremely powerful for branching and rejoining different layers. While Keras makes it easy to train complex models, I would like to dig more deep into some of the theory behind the different layers, and the reasoning behind various hyperparameters. Another thing that I learned was the difference between categorical cross entropy and binary cross entropy. I had originally just used categorical cross entropy for both the word prediction and category prediction.

This project also taught me the importance of starting early on projects. Both the preprocessing of the data and the training time resulted in many hours of debugging and waiting. The process of preparing the text data was challenging but can be easily used for other word based models I use in the future. I would definitely use a smaller window size for prediction and systematically increase it rather than jumping to something high. This would give me the opportunity to experiment more with other hyperparameters because it would reduce the amount of training time. Such as, increasing the dropout rate should make the model converge slower but overfit less often which I didn't give it the chance to do with such a small number of epochs.

### Why is it bad to validate/test on the training data?

It is bad to validate and test on the training data as we will tend to overfit when we eventually was to predict the test set. If we tweak the model too much while validating on the training data we will be not only overfitting in the model, but we will be human overfitting the data by modifying the model. Having a few hypothesis before hand and testing them out would be a better options
