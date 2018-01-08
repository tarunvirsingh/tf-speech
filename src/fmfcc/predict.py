import os
from glob import glob

import numpy as np
from tqdm import tqdm

from tensorflow.contrib.learn.python.learn.learn_io.generator_io import generator_input_fn
from tensorflow.contrib.training import HParams
from tensorflow.contrib.learn import RunConfig

from lib import create_estimator, model_dir, POSSIBLE_LABELS, params, id2name, FINGERPRINT_KEY, getMfcc, getTransformedAudioLocal
from features import FeatureGenerator

featureGenerator = FeatureGenerator(params)

TEST_BATCH_SIZE=64
TEST_DATA_PATHS = glob('../../data/test/audio/*wav')

def test_data_generator():
    for path in TEST_DATA_PATHS:
        fname = os.path.basename(path)
        result = dict(fname=np.string_(fname))
        audio_options = dict(
            fname=path,
            desired_samples=16000,
            fg_vol=1,
            bg_data=[],
            bg_vol=0,
            clip_min=-1.0,
            clip_max=1.0,
            time_shift_samples=0,
        )
        result[FINGERPRINT_KEY]=getMfcc(getTransformedAudioLocal(**audio_options))
        yield result

test_input_fn = generator_input_fn(
    x=test_data_generator,
    batch_size=TEST_BATCH_SIZE, 
    shuffle=False, 
    num_epochs=1,
    queue_capacity= 10 * TEST_BATCH_SIZE, 
    num_threads=1,
)

model = create_estimator(
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
