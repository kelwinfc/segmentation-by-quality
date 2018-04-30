import os, sys
sys.path.append(os.path.abspath(os.getcwd()))

from models.transformations import primitives
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import os
import pandas as pd
import json

class Transformer:
    def __init__(self, dataset, transformation):
        self.fn = getattr(primitives, transformation)
        filename = 'data/pdf/%s/%s.csv' % (dataset, transformation)
        self.pdf = pd.read_csv(filename, quotechar="'")

        # informative messages
        active = self.pdf.columns[(self.pdf > 0.001).iloc[0]]
        print('%s - There are %d active parameters out of %d parameters' % (
            transformation, len(active), len(self.pdf.columns)))
        print('Active parameters:', active)

    def __repr__(self):
        return self.fn.__name__

    def transform(self, mask, q, random_state):
        params = self.pdf.columns
        pdf = self.pdf.as_matrix()[0]

        ixnan = ~np.isnan(pdf)
        params = params[ixnan]
        pdf = pdf[ixnan]

        param = random_state.choice(params, p=pdf)
        param = json.loads(param)
        return self.fn(mask, random_state=random_state, **param)

if __name__ == '__main__':  # test if uniform
    import sys
    sys.path.append(os.path.abspath(os.getcwd()))
    from metrics.dice import dice_coef
    import matplotlib.pyplot as plt

    DATADIR = 'data'
    IMG_SHAPE = (128, 128)

    transfs = [
        'elastic_transform', 'flip_transform', 'morphological_transform',
        'switch_transform', 'rotate_transform', 'offset_transform',
    ]
    transfs = [Transformer(t) for t in transfs]
    filenames = os.listdir(DATADIR)

    for t in transfs:
        print(t)
        dices = []
        maxit = 10000
        for it in range(maxit):
            sys.stdout.write('\r%d%%' % (100*it/maxit))
            filename = np.random.choice(filenames)
            mask = imread(os.path.join(DATADIR, filename))
            mask = resize(mask, IMG_SHAPE, mode='reflect')
    
            q = np.random.random()
            mask2 = t.transform(mask, q)
            dices.append(dice_coef(mask, mask2))
        sys.stdout.write('\r                 \r')
    
        plt.hist(dices)
        plt.title(t)
        plt.show()
