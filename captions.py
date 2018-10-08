import mycoco

mycoco.setmode('train')

capt_iter = mycoco.iter_captions_cats(['horse', 'dog'])
print(next(capt_iter))
