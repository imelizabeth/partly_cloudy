#### Imports
import os
import numpy as np
import fdr_correction_helpers as helpers

### Set data to threshold
# Modify this section to change subject group
subject_group_str = '5-yrs'
print('Subject group:', subject_group_str)

# Modify this section to set score to correct
score_str = 'valence_difference_adult_isc_mask'
print('Scores to FDR correct:', score_str)

#### Load data
directory = os.path.join(os.path.expanduser('~'), 'data-lisik3', 'partly_cloudy', 'data', subject_group_str, 'encoding_results')
scores = np.load(os.path.join(directory, 'unthresholded', (subject_group_str + '_' + score_str + '.npy')))

#### Perform FDR correction
null_dist_path = os.path.join(directory, 'null_dists', (score_str + '_null_dists'))
null_dists = helpers.concatenate_null_dists(null_dist_path)
print('null dists:', np.shape(null_dists))
p = helpers.get_p_values(scores, null_dists)
print('p:', np.shape(p))
voxels, num_significant = helpers.get_fdr_controlled(p, threshold=0.05, observed=scores)

print(num_significant, 'significant voxels at threshold 0.05')

### Save results
save_path = os.path.join(directory, 'thresholded', (subject_group_str + '_' + score_str + '_thresholded05.npy'))
np.save(save_path, voxels)
np.save(os.path.join(directory, 'thresholded', (subject_group_str + '_' + score_str + '_p_values.npy')), p)
print('Successfully saved thresholded scores and p values')