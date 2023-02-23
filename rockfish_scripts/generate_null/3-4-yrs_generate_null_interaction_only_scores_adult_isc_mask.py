#### Imports

import os
import numpy as np
from fdr_correction_helpers import generate_null

#### Load scores

# Modify this section to change subject group
subject_group_str = '3-4-yrs'
print('Subject group:', subject_group_str)

# Modify this section to set score to generate null for
score_str = 'interaction_only_scores_adult_isc_mask'

# Load data
directory = os.path.join(os.path.expanduser('~'), 'data-lisik3', 'partly_cloudy', 'data', subject_group_str, 'encoding_results')
scores = np.load(os.path.join(directory, 'unthresholded', (subject_group_str + '_' + score_str + '.npy')))
print('Scores:', np.shape(scores))


#### Generate null distribution section
null_dist = generate_null(scores, n_permutations=100)
print('Null dist generated:', np.shape(null_dist))

#### Save null distribution section
dist_id = np.random.randint(100000)
dist_label = subject_group_str + '_' + score_str + '_null_dist' + str(dist_id) + '.npy'
np.save(os.path.join(directory, 'null_dists', (score_str + '_null_dists'), dist_label), null_dist)
print('Saved distribution to ', dist_label)