
- A diagram one of your architectures (probably, your best).

- A description of its architecture, design decisions, hyper-parameters (loss function, batch size, epochs, data size, etc).  You should also include a very brief summary of what you tried as a paragraph about the "big picture."

- Describe two more variants you tried:
  - One architectural variant (i.e., that changes the network graph).
  - Choose a hyperparameter for your "main" variant and pick two other variants of it (e.g. dropout, batch size, etc).  Describe why you picked that variation and what you expected.
  - Plot the losses (whichever loss function you're using) for your "main" architecture, the architectural variant, and the two hyperparameter variants, so four loss graphs.

- Describe in 1-2 paragraphs what you learned.

- To develop your "scientific common sense": Answer the question: why is it bad to validate/test on the training data?

- Include all the four saved model files in the repo in a manner that can be loaded by the testing script.

- If they are too large for github, we will come up with an alternative solution to provide them to us -- contact Vlad if so, before the submission deadline.

## Pre-Report

First we need to check out which of the models performed the best.


```python
import matplotlib.pyplot as plt

import pandas as pd

model_simple = pd.read_csv('./results/compare_simple.csv').head(10)
model_simple_embed100 = pd.read_csv('./results/compare_embed_100.csv').head(10)
model_complex_drop1 = pd.read_csv('./results/compare_complex_embed_100_do1.csv').head(10)
model_complex_drop5 = pd.read_csv('./results/compare_complex_embed_100_do5.csv').head(10)

# Plot overall loss
plt.plot(model_simple['epoch'], model_simple['loss'], label='Simple LSTM 50 Embed')
plt.plot(model_simple_embed100['epoch'], model_simple_embed100['loss'], label='Simple LSTM 100 Embed')
plt.plot(model_complex_drop1['epoch'], model_complex_drop1['loss'], label='Complex LSTM 100 Embed Dropout 0.1')
plt.plot(model_complex_drop5['epoch'], model_complex_drop5['loss'], label='Complex LSTM 100 Embed Dropout 0.5')

plt.legend(loc='upper right')
plt.title("Overall Categorical Cross Entropy Loss")

plt.savefig('./imgs/model_loss.png', bbox_inches='tight')
```


```python
# Plot word prediction loss
plt.plot(model_simple['epoch'], model_simple['word_prediction_loss'], label='Simple LSTM 50 Embed')
plt.plot(model_simple_embed100['epoch'], model_simple_embed100['word_prediction_loss'], label='Simple LSTM 100 Embed')
plt.plot(model_complex_drop1['epoch'], model_complex_drop1['word_prediction_loss'], label='Complex LSTM 100 Embed Dropout 0.1')
plt.plot(model_complex_drop5['epoch'], model_complex_drop5['word_prediction_loss'], label='Complex LSTM 100 Embed Dropout 0.5')

plt.legend(loc='upper right')
plt.title("Word Prediction Categorical Cross Entropy Loss")
```




    <matplotlib.text.Text at 0x119dd9a90>




```python
# Plot word prediction loss
plt.plot(model_simple['epoch'], model_simple['category_prediction_loss'], label='Simple LSTM 50 Embed')
plt.plot(model_simple_embed100['epoch'], model_simple_embed100['category_prediction_loss'], label='Simple LSTM 100 Embed')
plt.plot(model_complex_drop1['epoch'], model_complex_drop1['category_prediction_loss'], label='Complex LSTM 100 Embed Dropout 0.1')
plt.plot(model_complex_drop5['epoch'], model_complex_drop5['category_prediction_loss'], label='Complex LSTM 100 Embed Dropout 0.5')

plt.legend(loc='upper right')
plt.title("Category Prediction Categorical Cross Entropy Loss")
```




    <matplotlib.text.Text at 0x119dd9a90>



# LT2316 H18 Assignment 2

###  Model Diagram


```python
from keras.models import load_model
from keras.utils import plot_model

model = load_model('./models/lstm_simple_embed100.hdf5')
plot_model(model, to_file="./imgs/lstm_simple_embed100.jpg", show_shapes=True)
```

    Using TensorFlow backend.


![Simple](imgs/lstm_simple_embed100.jpg)

### Architecture Overview

After experiementing with many different architectures and LSTM layers, what ended up working the best was a single LSTM layer with 50 nodes. Each word is represented as an integer less than 10000. Some of the other approaches I tried were to have the embedding layer input into two different LSTM layers which in turn seperately predicted either word prediction, or category prediction. This performed worse than other options as well. The other architectures I experimented with are described below, where two LSTM layers are used (passing sequences from first to the second). 

  - **Loss function**: Categorical Cross Entropy
  - **Batch Size**: 256
  - **Epochs**: 10 (For final test)
  - **Window size**: 10 This was probably way overkill for this. I had slightly better results using 10 window size but the different in training time was not worth the results.
  - **Data size**: Using the window size of 10, I created 5572508 training examples from the 591753 captions in the training data. In the test set, 236352 training examples were created from 25014 captions.
  - **Dropout**: dropout was set to 0.1 for the LSTM layer. This performed better than no dropout and 0.5 dropout.


```python
# Alternative model
model = load_model('./models/lstm_complex_drop1.hdf5')
plot_model(model, to_file="./imgs/lstm_complex_drop1.jpg", show_shapes=True)
```

### Variant
![Two LSTM Layers](./imgs/lstm_complex_drop1.jpg)

This architectural variant did not seem to perform as well at the simple one layer LSTM model. I experimented with setting the dropout rate which increased the loss. This might not have been fair to do, since I did not run it for many epochs. Increasing the dropout rate should make the model converge slower but overfit less often which I didn't give it the chance to do with such a small number of epochs.

### Hyperparameter Selection  
  
  After experiementing with many different architectures and LSTM layers, what ended up working the best was a single LSTM layer with 50 nodes. I first changed the embedding size to 100 and then for the two other variations a more complex model (seen above) with additional dropout at levels 0.1 and 0.5. I thought that adding another LSTM layer would perform better than the single layer. I am not sure if increasing the epochs or modifying the other hyperparameters would have made a difference.
  
### Model Loss for Different Hyperparameters
  
![Model Loss](./imgs/model_loss.png)

### Lessons Learned

I learned lots about language models from building this word and category prediction. Also this project was a nice introduction to the Keras functional API. It was interesting to experiment with branching into two seperate LSTM models and combining at the end. The functional API is very simple to use and extremely powerful for branching and rejoining different layers. While Keras makes it easy to train complex models, I would like to dig more deep into some of the theory behind the different layers, and the reasoning behind various hyperparameters. 

This project also taught me the importance of starting early on projects. Both the preprocessing of the data and the training time resulted in many hours of debugging and waiting. The process of preparing the text data was challenging but can be easily used for other word based models I use in the future. I would definitely use a smaller window size for prediction and systematically increase it rather than jumping to something high. This would give me the opportunity to experiment more with other hyperparameters because it would reduce the amount of training time. Such as, increasing the dropout rate should make the model converge slower but overfit less often which I didn't give it the chance to do with such a small number of epochs.

### Why is it bad to validate/test on the training data?

It is bad to validate and test on the training data as we will tend to overfit when we eventually was to predict the test set. If we tweak the model too much while validating on the training data we will be not only overfitting in the model, but we will be human overfitting the data by modifying the model. Having a few hypothesis before hand and testing them out would be a better options
