#### Imports

import os
import numpy as np

subject_groups = ['3-4-yrs', '5-yrs', '7-yrs', '8-12-yrs']
directory = os.path.join(os.path.expanduser('~'), 'data-lisik3', 'partly_cloudy', 'data')

# MOTEN SCORES
score_str = 'full_scores_adult_isc_mask'
save_str = 'full_difference_adult_isc_mask'

# Load data
adult_scores = np.load(os.path.join(directory, 'adults', 'encoding_results', 'unthresholded', ('adults_' + score_str + '.npy')))

for subject_group_str in subject_groups:
        # Load child data
        child_scores = np.load(os.path.join(directory, subject_group_str, 'encoding_results', 'unthresholded', (subject_group_str + '_' + score_str + '.npy')))

        # Save difference
        scores_difference = np.subtract(np.mean(adult_scores, axis=0), np.mean(child_scores, axis=0))
        np.save(os.path.join(directory, subject_group_str,'encoding_results', 'unthresholded', (subject_group_str + '_' + save_str + '.npy')), scores_difference)
        print('Saved score difference', np.shape(scores_difference), 'for subject group', subject_group_str)


