import os
from glob import glob

import numpy as np
from tqdm import tqdm

from tensorflow.contrib.learn.python.learn.learn_io.generator_io import generator_input_fn
from tensorflow.contrib.training import HParams
from tensorflow.contrib.learn import RunConfig

from lib import create_model, model_dir, POSSIBLE_LABELS, params, id2name, FINGERPRINT_KEY
from features import FeatureGenerator

featureGenerator = FeatureGenerator(params)

TEST_BATCH_SIZE=64
TEST_DATA_PATHS = glob('../../data/test/audio/*wav')

def test_data_generator():
    for path in TEST_DATA_PATHS:
        fname = os.path.basename(path)
        result = dict(fname=np.string_(fname))
        result[FINGERPRINT_KEY]=featureGenerator.getFeatures(filename=path)
        yield result

test_input_fn = generator_input_fn(
    x=test_data_generator,
    batch_size=TEST_BATCH_SIZE, 
    shuffle=False, 
    num_epochs=1,
    queue_capacity= 10 * TEST_BATCH_SIZE, 
    num_threads=1,
)

model = create_model(
            config=RunConfig(model_dir=model_dir), 
            hparams=HParams(**params),
        )
it = model.predict(input_fn=test_input_fn)

submission = dict()
for t in tqdm(it):
    fname, label = t['fname'].decode(), id2name[t['label']]
    submission[fname] = label

with open(os.path.join(model_dir, 'submission.csv'), 'w') as fout:
    fout.write('fname,label\n')
    for fname, label in submission.items():
        fout.write('{},{}\n'.format(fname, label))
