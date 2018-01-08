import os
from glob import glob
import math

import numpy as np
from scipy.io import wavfile

from tensorflow.contrib.learn import RunConfig, Experiment, learn_runner
from tensorflow.contrib.learn.python.learn.learn_io.generator_io import generator_input_fn
from tensorflow.contrib.training import HParams

from lib import create_estimator, model_dir, POSSIBLE_LABELS, params, id2name, FINGERPRINT_KEY, getMfcc, getTransformedAudioLocal
from features import FeatureGenerator

DATADIR = '../../data'
POSSIBLE_LABELS_SET = set(POSSIBLE_LABELS)
name2id = {name: i for i, name in id2name.items()}
BATCH_SIZE=64
TARGET_KEY='target'
SILENCE_PCT=10
BG_PARAMS={
    'bg_freq': 0.8,
    'bg_vol_range': 0.1,
}
TIME_SHIFT_MS=100.0
TIME_SHIFT_SAMPLES = int((TIME_SHIFT_MS * 16000) / 1000)


def get_metadata_lists(data_dir):
    all_files = glob(os.path.join(data_dir, 'train/audio/*/*wav'))
    # all_files = [ x.split('audio/')[1] for x in all_files if not '_background_noise_' in x ]
    # all_files = [ x for x in all_files[:10] ]
    val_users = [];
    with open(os.path.join(data_dir, 'train/validation_list.txt'), 'r') as fin:
        validation_files = fin.readlines()
        val_users = { getUid(x) for x in validation_files }

    train, val = [], []
    for filePath in all_files:
        partialPath = filePath.split('audio/')[1]
        uid = getUid(partialPath)
        label = getLabel(partialPath)
        if label == '_background_noise_':
            continue
        if label not in POSSIBLE_LABELS_SET:
            label = 'unknown'
        sample = (label, filePath)
        if uid in val_users:
            val.append(sample)
        else:
            train.append(sample)

    print('There are {} train and {} val samples'.format(len(train), len(val)))
    return train, val

def getBgFileNames(data_dir):
    return glob(os.path.join(data_dir, 'train/audio/_background_noise_/*wav'))

def getUid(filePath):
    return filePath.split('/')[1].split('_')[0];

def getLabelId(filePath):
    label = filePath.split('/')[0]
    if label == '_background_noise_':
        label = 'silence'
    if label not in POSSIBLE_LABELS_SET:
        label = 'unknown'
    label_id = name2id[label]
    return label_id

def getLabel(filePath):
    return filePath.split('/')[0]

def data_generator_fn(meta_list, bg_params, mode=None):
    def generator():
        np.random.shuffle(meta_list)
        # if mode == 'train':
            # np.random.shuffle(meta_list)
        for (label, fname) in meta_list:
            result = dict();
            audio_options = dict(
                fname=fname,
                desired_samples=16000,
                fg_vol=0 if label == 'silence' else 1,
                bg_data=bg_data,
                bg_vol=getBgVol(bg_params['bg_freq'], bg_params['bg_vol_range']),
                clip_min=-1.0,
                clip_max=1.0,
                time_shift_samples=TIME_SHIFT_SAMPLES,
            )
            result[FINGERPRINT_KEY]=getMfcc(getTransformedAudioLocal(**audio_options))
            result[TARGET_KEY]=np.int32(name2id[label])
            yield result
    return generator

def augmentWithSilence(meta_list, silencePct):
    someFName = meta_list[0][1] # need any valid audio file. Will be mult by 0
    silence_size = int(math.ceil(len(meta_list) * silencePct/ 100))
    for _ in range(silence_size):
        meta_list.append(('silence', someFName))

def getBgVol(background_frequency, background_volume_range):
    if np.random.uniform(0, 1) < background_frequency:
      return np.random.uniform(0, background_volume_range)
    else:
      return 0

# print('bck vol: ', getBckVol(0.5, 0.5))
# print('bck vol: ', getBckVol(0.5, 0.5))
# print('bck vol: ', getBckVol(0.5, 0.5))
# print('bck vol: ', getBckVol(0.5, 0.5))

def getAudioLocal(fname):
    _, wav = wavfile.read(fname)
    wav = wav.astype(np.float32) / np.iinfo(np.int16).max
    return wav

##=========================================================
## Actual computations start here
##=========================================================
bg_data = [ getAudioLocal(x) for x in getBgFileNames(DATADIR) ]

train_meta_list, val_meta_list= get_metadata_lists(DATADIR)
augmentWithSilence(train_meta_list, SILENCE_PCT)
augmentWithSilence(val_meta_list, SILENCE_PCT)
print('Augumented sizes: Train: {}. Val: {}'.format(len(train_meta_list), len(val_meta_list)))
train_input_fn = generator_input_fn(
    x=data_generator_fn(train_meta_list, BG_PARAMS, 'train'),
    target_key=TARGET_KEY,
    batch_size=BATCH_SIZE, shuffle=True, num_epochs=None,
    queue_capacity=3 * BATCH_SIZE + 10, num_threads=1,
)
val_input_fn = generator_input_fn(
    x=data_generator_fn(val_meta_list, BG_PARAMS, 'eval'),
    target_key=TARGET_KEY,
    batch_size=BATCH_SIZE, shuffle=True, num_epochs=None,
    queue_capacity=3 * BATCH_SIZE + 10, num_threads=1,
)
def experiment_fn(run_config, hparams):
    return Experiment(
        estimator=create_estimator(config=run_config, hparams=hparams),
        train_input_fn=train_input_fn,
        eval_input_fn=val_input_fn,
        train_steps=10000,
        eval_delay_secs=1,
        eval_steps=200,
        train_steps_per_iteration=1000,
    )
learn_runner.run(
    experiment_fn=experiment_fn,
    run_config=RunConfig(model_dir=model_dir),
    schedule="continuous_train_and_eval",
    # schedule="evaluate",
    hparams=HParams(**params))
