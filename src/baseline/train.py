import os
from glob import glob

import numpy as np
from scipy.io import wavfile

from tensorflow.contrib.learn import RunConfig, Experiment, learn_runner
from tensorflow.contrib.learn.python.learn.learn_io.generator_io import generator_input_fn
from tensorflow.contrib.training import HParams

from lib import create_model, model_dir, POSSIBLE_LABELS, params, id2name

DATADIR = '../../data'
POSSIBLE_LABELS_SET = set(POSSIBLE_LABELS)
name2id = {name: i for i, name in id2name.items()}
BATCH_SIZE=64
TARGET_KEY='target'

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
        sample = (getLabelId(partialPath), uid, filePath)
        if uid in val_users:
            val.append(sample)
        else:
            train.append(sample)

    print('There are {} train and {} val samples'.format(len(train), len(val)))
    return train, val

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

def data_generator_fn(data, mode=None):
    def generator():
        if mode == 'train':
            np.random.shuffle(data)
        for (label_id, uid, fname) in data:
            try:
                _, wav = wavfile.read(fname)
                wav = wav.astype(np.float32) / np.iinfo(np.int16).max

                L = 16000  # be aware, some files are shorter than 1 sec!
                if len(wav) < L:
                    continue
                # let's generate more silence!
                samples_per_file = 1 if label_id != name2id['silence'] else 20
                for _ in range(samples_per_file):
                    if len(wav) > L:
                        beg = np.random.randint(0, len(wav) - L)
                    else:
                        beg = 0
                    result = dict(wav=wav[beg: beg + L])
                    result[TARGET_KEY]=np.int32(label_id)
                    yield result
            except Exception as err:
                print(err, label_id, uid, fname)
    return generator

##=========================================================
## Actual computations start here
##=========================================================
train_meta_list, val_meta_list= get_metadata_lists(DATADIR)
train_input_fn = generator_input_fn(
    x=data_generator_fn(train_meta_list, 'train'),
    target_key=TARGET_KEY,
    batch_size=BATCH_SIZE, shuffle=True, num_epochs=None,
    queue_capacity=3 * BATCH_SIZE + 10, num_threads=1,
)
val_input_fn = generator_input_fn(
    x=data_generator_fn(val_meta_list),
    target_key=TARGET_KEY,
    batch_size=BATCH_SIZE, shuffle=True, num_epochs=None,
    queue_capacity=3 * BATCH_SIZE + 10, num_threads=1,
)
def experiment_fn(run_config, hparams):
    return Experiment(
        estimator=create_model(config=run_config, hparams=hparams),
        train_input_fn=train_input_fn,
        eval_input_fn=val_input_fn,
        train_steps=10000,
        eval_steps=200,
        train_steps_per_iteration=1000,
    )
# os.makedirs(os.path.join(MODEL_DIR, 'eval'), exist_ok=True)
learn_runner.run(
    experiment_fn=experiment_fn,
    run_config=RunConfig(model_dir=model_dir),
    schedule="continuous_train_and_eval",
    hparams=HParams(**params))
