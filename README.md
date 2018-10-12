# LT2316 H18 Assignment 2

Git project for implementing assignment 2 in [Asad Sayeed's](https://asayeed.github.io) machine learning class in the University of Gothenburg's Masters
of Language Technology programme.

## Word Prediction
```
python predict_word.py --windowsize=10 "A man standing on top of a" /scratch/gussteen/lstm_simple.12.hdf5 tokenizer10000.pickle --npredictions=3
```

## Testing
```
python test.py -P B ./models/lstm_complex_embed100_drop1.hdf5 tokenizer10000.pickle --maxinstances=50
```

## Training
Example
```
python train.py -P B --init_model=init_model.hdf5 /scratch/gussteen/ test_model.hdf5 ['dog', 'horse']
```

## Report
  - A diagram one of your architectures (probably, your best).
  - A description of its architecture, design decisions, hyper-parameters (loss function, batch size, epochs, data size, etc).  You should also include a very brief summary of what you tried as a paragraph about the "big picture."
  - Describe two more variants you tried:
    - One architectural variant (i.e., that changes the network graph).
    - Choose a hyperparameter for your "main" variant and pick two other variants of it (e.g. dropout, batch size, etc).  Describe why you picked that variation and what you expected.
    - Plot the losses (whichever loss function you're using) for your "main" architecture, the architectural variant, and the two hyperparameter variants, so four loss graphs.
  - Describe in 1-2 paragraphs what you learned.
  - To develop your "scientific common sense": Answer the question: why is it bad to validate/test on the training data?
  - Include all the four saved model files in the repo in a manner that can be loaded by the testing script. If they are too large for github, we will come up with an alternative solution to provide them to us -- contact Vlad if so, before the submission deadline.
