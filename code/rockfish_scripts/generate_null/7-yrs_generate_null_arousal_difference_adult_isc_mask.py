#### Imports

import os
import numpy as np
from fdr_correction_helpers import generate_null_comparison

#### Load scores

# Modify this section to change subject group
subject_group_str = '7-yrs'
print('Subject group:', subject_group_str)

# Modify this section to set score to generate null for
score_str = 'arousal_only_scores_adult_isc_mask'
save_str = 'arousal_difference_adult_isc_mask'

# Load data

directory = os.path.join(os.path.expanduser('~'), 'data-lisik3', 'partly_cloudy', 'data')

child_scores = np.load(os.path.join(directory, subject_group_str, 'encoding_results', 'unthresholded', (subject_group_str + '_' + score_str + '.npy')))
adult_scores = np.load(os.path.join(directory, 'adults', 'encoding_results', 'unthresholded', ('adults_' + score_str + '.npy')))

#### Generate null distribution section
null_dist = generate_null_comparison(adult_scores, child_scores, n_permutations=100)
print('Null dist generated:', np.shape(null_dist))

#### Save null distribution section
dist_id = np.random.randint(100000)
dist_label = subject_group_str + '_' + save_str + '_null_dist' + str(dist_id) + '.npy'
np.save(os.path.join(directory, subject_group_str,'encoding_results', 'null_dists', (save_str + '_null_dists'), dist_label), null_dist)
print('Saved distribution to ', dist_label)