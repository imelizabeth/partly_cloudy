#### Imports

import os
import numpy as np

subject_groups = ['3-4-yrs', '5-yrs', '7-yrs', '8-12-yrs']
directory = os.path.join(os.path.expanduser('~'), 'data-lisik3', 'partly_cloudy', 'data')

# SOCIAL SCORES
score_str = 'social_only_scores_adult_isc_mask'
save_str = 'social_difference_adult_isc_mask'

# Load data
adult_scores = np.load(os.path.join(directory, 'adults', 'encoding_results', 'unthresholded', ('adults_' + score_str + '.npy')))

for subject_group_str in subject_groups:
        # Load child data
        child_scores = np.load(os.path.join(directory, subject_group_str, 'encoding_results', 'unthresholded', (subject_group_str + '_' + score_str + '.npy')))

        # Save difference
        scores_difference = np.subtract(np.mean(adult_scores, axis=0), np.mean(child_scores, axis=0))
        np.save(os.path.join(directory, subject_group_str,'encoding_results', 'unthresholded', (subject_group_str + '_' + save_str + '.npy')), scores_difference)
        print('Saved score difference', np.shape(scores_difference), 'for subject group', subject_group_str)

# AROUSAL SCORES
score_str = 'arousal_only_scores_adult_isc_mask'
save_str = 'arousal_difference_adult_isc_mask'

# Load data
adult_scores = np.load(os.path.join(directory, 'adults', 'encoding_results', 'unthresholded', ('adults_' + score_str + '.npy')))

for subject_group_str in subject_groups:
        # Load child data
        child_scores = np.load(os.path.join(directory, subject_group_str, 'encoding_results', 'unthresholded', (subject_group_str + '_' + score_str + '.npy')))

        # Save difference
        scores_difference = np.subtract(np.mean(adult_scores, axis=0), np.mean(child_scores, axis=0))
        np.save(os.path.join(directory, subject_group_str,'encoding_results', 'unthresholded', (subject_group_str + '_' + save_str + '.npy')), scores_difference)
        print('Saved score difference', np.shape(scores_difference), 'for subject group', subject_group_str)

# FACES SCORES
score_str = 'faces_only_scores_adult_isc_mask'
save_str = 'faces_difference_adult_isc_mask'

# Load data
adult_scores = np.load(os.path.join(directory, 'adults', 'encoding_results', 'unthresholded', ('adults_' + score_str + '.npy')))

for subject_group_str in subject_groups:
        # Load child data
        child_scores = np.load(os.path.join(directory, subject_group_str, 'encoding_results', 'unthresholded', (subject_group_str + '_' + score_str + '.npy')))

        # Save difference
        scores_difference = np.subtract(np.mean(adult_scores, axis=0), np.mean(child_scores, axis=0))
        np.save(os.path.join(directory, subject_group_str,'encoding_results', 'unthresholded', (subject_group_str + '_' + save_str + '.npy')), scores_difference)
        print('Saved score difference', np.shape(scores_difference), 'for subject group', subject_group_str)

# INTERACTION SCORES
score_str = 'interaction_only_scores_adult_isc_mask'
save_str = 'interaction_difference_adult_isc_mask'

# Load data
adult_scores = np.load(os.path.join(directory, 'adults', 'encoding_results', 'unthresholded', ('adults_' + score_str + '.npy')))

for subject_group_str in subject_groups:
        # Load child data
        child_scores = np.load(os.path.join(directory, subject_group_str, 'encoding_results', 'unthresholded', (subject_group_str + '_' + score_str + '.npy')))

        # Save difference
        scores_difference = np.subtract(np.mean(adult_scores, axis=0), np.mean(child_scores, axis=0))
        np.save(os.path.join(directory, subject_group_str,'encoding_results', 'unthresholded', (subject_group_str + '_' + save_str + '.npy')), scores_difference)
        print('Saved score difference', np.shape(scores_difference), 'for subject group', subject_group_str)


# TOM SCORES
score_str = 'tom_only_scores_adult_isc_mask'
save_str = 'tom_difference_adult_isc_mask'

# Load data
adult_scores = np.load(os.path.join(directory, 'adults', 'encoding_results', 'unthresholded', ('adults_' + score_str + '.npy')))

for subject_group_str in subject_groups:
        # Load child data
        child_scores = np.load(os.path.join(directory, subject_group_str, 'encoding_results', 'unthresholded', (subject_group_str + '_' + score_str + '.npy')))

        # Save difference
        scores_difference = np.subtract(np.mean(adult_scores, axis=0), np.mean(child_scores, axis=0))
        np.save(os.path.join(directory, subject_group_str,'encoding_results', 'unthresholded', (subject_group_str + '_' + save_str + '.npy')), scores_difference)
        print('Saved score difference', np.shape(scores_difference), 'for subject group', subject_group_str)


# VALENCE SCORES
score_str = 'valence_only_scores_adult_isc_mask'
save_str = 'valence_difference_adult_isc_mask'

# Load data
adult_scores = np.load(os.path.join(directory, 'adults', 'encoding_results', 'unthresholded', ('adults_' + score_str + '.npy')))

for subject_group_str in subject_groups:
        # Load child data
        child_scores = np.load(os.path.join(directory, subject_group_str, 'encoding_results', 'unthresholded', (subject_group_str + '_' + score_str + '.npy')))

        # Save difference
        scores_difference = np.subtract(np.mean(adult_scores, axis=0), np.mean(child_scores, axis=0))
        np.save(os.path.join(directory, subject_group_str,'encoding_results', 'unthresholded', (subject_group_str + '_' + save_str + '.npy')), scores_difference)
        print('Saved score difference', np.shape(scores_difference), 'for subject group', subject_group_str)



