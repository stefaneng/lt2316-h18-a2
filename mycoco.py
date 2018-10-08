# This is a helper module that contains conveniences to access the MS COCO
# dataset. You can modify at will.  In fact, you will almost certainly have
# to, or implement otherwise.

import sys
import itertools
import json
import pickle
import os.path

# This is evil, forgive me, but practical under the circumstances.
# It's a hardcoded access to the COCO API.  
COCOAPI_PATH='/scratch/lt2316-h18-resources/cocoapi/PythonAPI/'
TRAIN_ANNOT_FILE='/scratch/lt2316-h18-resources/coco/annotations/instances_train2017.json'
VAL_ANNOT_FILE='/scratch/lt2316-h18-resources/coco/annotations/instances_val2017.json'
TRAIN_CAP_FILE='/scratch/lt2316-h18-resources/coco/annotations/captions_train2017.json'
VAL_CAP_FILE='/scratch/lt2316-h18-resources/coco/annotations/captions_val2017.json'
TRAIN_IMG_DIR='/scratch/lt2316-h18-resources/coco/train2017/'
VAL_IMG_DIR='/scratch/lt2316-h18-resources/coco/val2017/'
annotfile = TRAIN_ANNOT_FILE
capfile = TRAIN_CAP_FILE
imgdir = TRAIN_IMG_DIR

sys.path.append(COCOAPI_PATH)
from pycocotools.coco import COCO

annotcoco = None
capcoco = None

# OK back to normal.
import random
import skimage.io as io
import skimage.transform as tform
import numpy as np

def setmode(mode):
    '''
    Set entire module's mode as 'train' or 'test' for the purpose of data extraction.
    '''
    global annotfile
    global capfile
    global imgdir
    global annotcoco, capcoco
    if mode == "train":
        annotfile = TRAIN_ANNOT_FILE
        capfile = TRAIN_CAP_FILE
        imgdir = TRAIN_IMG_DIR
    elif mode == "test":
        annotfile = VAL_ANNOT_FILE
        capfile = VAL_CAP_FILE
        imgdir = VAL_IMG_DIR
    else:
        raise ValueError

    annotcoco = COCO(annotfile)
    capcoco = COCO(capfile)

    
def query(queries, exclusive=True):
    '''
    Collects mutually-exclusive lists of COCO ids by queries, so returns 
    a parallel list of lists.
    (Setting 'exclusive' to False makes the lists non-exclusive.)  
    e.g., exclusive_query([['toilet', 'boat'], ['umbrella', 'bench']])
    to find two mutually exclusive lists of images, one with toilets and
    boats, and the other with umbrellas and benches in the same image.
    '''
    if not annotcoco:
        raise ValueError
    imgsets = [set(annotcoco.getImgIds(catIds=annotcoco.getCatIds(catNms=x))) for x in queries]
    if len(queries) > 1:
        if exclusive:
            common = set.intersection(*imgsets)
            return [[x for x in y if x not in common] for y in imgsets]
        else:
            return [list(y) for y in imgsets]
    else:
        return [list(imgsets[0])]
    
def iter_captions_cats(cats, intersection=False):
    '''
    `cats` is a 1d array of categories
    `exclusive` is a boolean that indicates whether the images should contain ALL categories in `cat` or 
                return images that contain at least one of the captions.
    Iterates over captions with includes the other captions associated with the image (excluding the ones given in cats)    
    '''
    # Manually access the json since I couldn't figure out how to do this with COCOAPI
    # Could test if we need to recompute
    #     if not image_cat:
    #    global image_cat
    anno_json = None
    with open(TRAIN_ANNOT_FILE) as fa:
        anno_json = json.load(fa)

    # Create an image id: category dictionary that we save between runs
    # to save time
    image_cat = {}
    if os.path.isfile("image_categories.pickle"):
        with open('image_categories.pickle', 'rb') as f:
            image_cat = pickle.load(f)
    else:        
        # Create a dictionary with categories for each image
        for a in anno_json['annotations']:
            img_id = a['image_id']
            cat_id = a['category_id']
            if img_id not in image_cat:
                image_cat[img_id] = set()
            if cat_id not in image_cat[img_id]:
                image_cat[img_id].add(cat_id)
        
        # Write to file
        with open('image_categories.pickle', 'wb+') as f:
            pickle.dump(image_cat, f)
        
    # if exlusive: surround with []
    if intersection:
        cats = [cats]
    # Query non-exclusive so we get images from intersection as well
    query_res = query(cats, exclusive=False)
    # Join the results from the query into one set
    imageids = set(itertools.chain(* query_res))
   
    annids =  capcoco.getAnnIds(imgIds=imageids)
    anns = capcoco.loadAnns(annids)
    random_anns = random.sample(anns, k=len(anns))
    for a in random_anns:
        # Print out the name of the category, rather than 
        a['categories'] = [annotcoco.cats[imgid]['name'] for imgid in image_cat[a['image_id']]]
        yield a
    
    
def iter_captions(idlists, cats, batch=1):
    '''
    Obtains the corresponding captions from multiple COCO id lists.
    Randomizes the order.  
    Returns an infinite iterator (do not convert to list!) that returns tuples (captions, categories)
    as parallel lists at size of batch.
    '''
    if not capcoco:
        raise ValueError
    if batch < 1:
        raise ValueError

    full = []
    for z in zip(idlists, cats):
        for x in z[0]:
            full.append((x, z[1]))
        
    while True:
        randomlist = random.sample(full, k=len(full))
        captions = []
        labels = []

        for p in randomlist:
            annids =  capcoco.getAnnIds(imgIds=[p[0]])
            anns = capcoco.loadAnns(annids)
            for ann in anns:
                captions.append(ann['caption'])
                # For LSTM you may want to do more with the captions
                # or otherwise distribute the data.
                labels.append(p[1])
                if len(captions) % batch == 0:
                    yield (captions, labels)
                    captions = []
                    labels = []

def iter_images(idlists, cats, size=(200,200), batch=1):
    '''
    Obtains the corresponding image data as numpy array from multiple COCO id lists.
    Returns an infinite iterator (do not convert to list!) that returns tuples (imagess, categories)
    as parallel lists at size of batch.
    By default, randomizes the order and resizes the image.
    '''
    if not annotcoco:
        raise ValueError
    if batch < 1:
        raise ValueError
    if not size:
        raise ValueError # size is mandatory

    full = []
    for z in zip(idlists, cats):
        for x in z[0]:
            full.append((x, z[1]))

    while True:
        randomlist = random.sample(full, k=len(full))

        images = []
        labels = []
        for r in randomlist:
            imgfile = annotcoco.loadImgs([r[0]])[0]['file_name']
            img = io.imread(TRAIN_IMG_DIR + imgfile)
            imgscaled = tform.resize(img, size)
            # Colour images only.
            if imgscaled.shape == (size[0], size[1], 3):
                images.append(imgscaled)
                labels.append(r[1])
                if len(images) % batch == 0:
                    yield (np.array(images), np.array(labels))
                    images = []
                    labels = []
