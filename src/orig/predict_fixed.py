from glob import glob
import os

from tqdm import tqdm

def run(tests, fixed_prediction):
  result = []
  for test in tqdm(tests):
      result.append({ 'filename': test['filename'], 'label': fixed_prediction})
  return result

def getTest(filepath):
    return { 'filepath': filepath, 'filename': os.path.basename(filepath) }

fixed_prediction='unknown'
test_data_paths = glob('../../data/test/audio/*wav')
tests = [ getTest(x) for x in test_data_paths ]
results = run(tests, fixed_prediction)

submissionFileName='./model/submission_' + fixed_prediction + '.csv'
with open(submissionFileName, 'w') as fout:
    fout.write('fname,label\n')
    for result in results:
        fout.write('{},{}\n'.format(result['filename'], result['label']))
