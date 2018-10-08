import mycoco
import pickle

mycoco.setmode('train')

capt_iter = mycoco.iter_captions_cats(['horse', 'dog'])

print(next(capt_iter))
with open('horse_dog_sample.pickle', 'wb+') as f:
    pickle.dump(list(capt_iter), f)
