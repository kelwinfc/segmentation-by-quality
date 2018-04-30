from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import ParameterGrid
import primitives
import scipy.optimize
import numpy as np
import os
import shutil
import sys


if len(sys.argv) < 2:
    print("Usage: %s DATASET" % sys.argv[0])
    sys.exit(-1)

def dice_coef(y, yp):
    smooth = 1
    intersection = np.sum(y * yp)
    return (2 * intersection + smooth) / (np.sum(y) + np.sum(yp) + smooth)

DATASET = sys.argv[1]

DATADIR = '../data/partitions/' + DATASET
OUTDIR = '../data/pdf/' + DATASET
IMG_SHAPE = (128, 128)
NREPS = 10
NBINS = 8

if os.path.exists(OUTDIR):
    shutil.rmtree(OUTDIR)
os.mkdir(OUTDIR)

filenames = \
    [os.path.join(DATADIR, 'train', 'masks', 'seg', f) for f in os.listdir(os.path.join(DATADIR, 'train', 'masks', 'seg'))] + \
    [os.path.join(DATADIR, 'validation', 'masks', 'seg', f) for f in os.listdir(os.path.join(DATADIR, 'validation', 'masks', 'seg'))]
    #[os.path.join(DATADIR, 'test', 'masks', 'seg', f) for f in os.listdir(os.path.join(DATADIR, 'test', 'masks', 'seg'))]

transfs = (
    (primitives.elastic_transform, {'alpha': np.arange(0, 6, dtype=int), 'sigma': np.linspace(0, 0.10, 20, True, dtype=float), 'alpha_affine': np.linspace(0, 0.10, 20, True, dtype=float)}, NREPS),
    (primitives.morphological_transform, {'size': np.linspace(-20, 20, 20, True, dtype=int)}, 1),
    (primitives.flip_transform, {'flipdir': np.array((0, 1, 2))}, 1),
    (primitives.switch_transform, {'n': np.linspace(0, np.prod(IMG_SHAPE), 20, True, dtype=int)}, NREPS),
    (primitives.rotate_transform, {'angle': np.linspace(0, 360, 20, dtype=int)}, 1),
    (primitives.offset_transform, {'xoffset': np.linspace(-30, 30, 10, True, dtype=int), 'yoffset': np.linspace(-30, 30, 10, True, dtype=int)}, 1),
    (primitives.flip_offset_transform, {'xoffset': np.linspace(-30, 30, 10, True, dtype=int), 'yoffset': np.linspace(-30, 30, 10, True, dtype=int), 'flipdir': np.array((0, 1, 2))}, 1),
)
transfs = [(t, ParameterGrid(g), nreps) for t, g, nreps in transfs]

NBINS = 8
bins = np.linspace(0, 1, NBINS, False)[1:]
freqs = [np.zeros((NBINS, len(params)), int) for _, params, __ in transfs]

progress = 0
total_progress = sum(len(filenames) * len(transfs) * nreps * \
    len(g) for _, g, nreps in transfs)

for fi, filename in enumerate(filenames):
    mask = imread(filename)
    mask = resize(mask, IMG_SHAPE, mode='reflect')
    for j, (t, params, nreps) in enumerate(transfs):
        for pi, param in enumerate(params):
            for i in range(nreps):
                sys.stdout.write('\r%5.1f%% %-10s %-30s' % (
                    100*progress/total_progress, filename, t.__name__))
                mask2 = t(mask, **param, random_state=np.random)
                q = dice_coef(mask, mask2)
                bin = (q >= bins).sum()
                freqs[j][bin, pi] += 1
                progress += 1

sys.stdout.write('\r                                                       \r')

for freq, (t, params, _) in zip(freqs, transfs):
    pdf = freq / freq.sum(0)
    #p = np.linalg.lstsq(pdf, np.repeat(1/len(pdf), len(pdf)))[0]
    y = np.repeat(1/len(pdf), len(pdf))
    p = scipy.optimize.lsq_linear(pdf, y, bounds=(0,np.inf))['x']
    p = p/p.sum()
    header = ','.join("'%s'" % str(p).replace("'", '"') for p in params)
    filename = '%s.csv' % t.__name__
    with open(os.path.join(OUTDIR, filename), 'w') as f:
        f.write(header + '\n')
        f.write(','.join(map(str, p)))
