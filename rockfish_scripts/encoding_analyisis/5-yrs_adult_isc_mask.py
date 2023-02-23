#### Imports
import os
import numpy as np
import encoding_helpers as helpers

#### Modify this section to change subject group
subject_group_str = '5-yrs'
mask_str = 'adult_isc_mask'
print('Subject group:', subject_group_str)

#### Load data
directory = os.path.join(os.path.expanduser('~'), 'data-lisik3',  'partly_cloudy', 'data')

# Load BOLD data
bold_data = np.load(os.path.join(directory, subject_group_str, (subject_group_str + '_normalized_data_' + mask_str + '.npy')))

# Load features
social_features = np.load(os.path.join(directory, 'features', 'social_normalized.npy'))
moten_features = np.load(os.path.join(directory, 'features', 'moten_reduced19.npy'))

# Combine features for use in model predictions
combined_features = np.concatenate((social_features, moten_features), axis=1)

#### Generate test / train split for features

from sklearn import model_selection

social_train, social_test, moten_train, moten_test, combined_train, combined_test = model_selection.train_test_split(
        social_features, moten_features, combined_features, test_size=0.2, random_state=4)


#### Generate encoding weights

weights = []
full_prediction_scores = []
social_only_prediction_scores = []
moten_only_prediction_scores = []
faces_only_prediction_scores = []
interaction_only_prediction_scores = []
tom_only_prediction_scores = []
valence_only_prediction_scores = []
arousal_only_prediction_scores = []

for subject in bold_data:

        # Reorient subject data so axes align with feature axes
        subject_data = np.swapaxes(subject, 0, 1)

        # Generate test / train split for data -- RANDOM
        subject_train, subject_test = model_selection.train_test_split(subject_data, test_size=0.2, random_state=4)

        # Generate encoding weights
        subject_weights = helpers.get_banded_weights([social_features, moten_features],
                                             [social_train, moten_train],
                                             [social_test, moten_test],
                                             subject_train, subject_test)
        weights.append(subject_weights)

        # Get and score predictions -- full weights
        subject_predictions = helpers.get_predictions_from_weights(combined_test, subject_weights)
        subject_prediction_scores = helpers.get_prediction_scores(subject_predictions, subject_test)
        full_prediction_scores.append(subject_prediction_scores)

        # Get and score predictions -- social only weights
        subject_social_only_weights = helpers.isolate_feature(subject_weights, start_index=0, end_index=5)
        subject_social_predictions = helpers.get_predictions_from_weights(combined_test, subject_social_only_weights)
        subject_social_prediction_scores = helpers.get_prediction_scores(subject_social_predictions, subject_test)
        social_only_prediction_scores.append(subject_social_prediction_scores)

        # Get and score predictions -- moten only weights
        subject_moten_only_weights = helpers.isolate_feature(subject_weights, start_index=5, end_index=len(combined_features))
        subject_moten_predictions = helpers.get_predictions_from_weights(combined_test, subject_moten_only_weights)
        subject_moten_prediction_scores = helpers.get_prediction_scores(subject_moten_predictions, subject_test)
        moten_only_prediction_scores.append(subject_moten_prediction_scores)

        # Get and score predictions -- faces only weights
        subject_faces_only_weights = helpers.isolate_feature(subject_weights, start_index=0, end_index=1)
        subject_faces_predictions = helpers.get_predictions_from_weights(combined_test, subject_faces_only_weights)
        subject_faces_prediction_scores = helpers.get_prediction_scores(subject_faces_predictions, subject_test)
        faces_only_prediction_scores.append(subject_faces_prediction_scores)

        # Get and score predictions -- interaction only weights
        subject_interaction_only_weights = helpers.isolate_feature(subject_weights, start_index=1, end_index=2)
        subject_interaction_predictions = helpers.get_predictions_from_weights(combined_test, subject_interaction_only_weights)
        subject_interaction_prediction_scores = helpers.get_prediction_scores(subject_interaction_predictions, subject_test)
        interaction_only_prediction_scores.append(subject_interaction_prediction_scores)

        # Get and score predictions -- ToM only weights
        subject_tom_only_weights = helpers.isolate_feature(subject_weights, start_index=2, end_index=3)
        subject_tom_predictions = helpers.get_predictions_from_weights(combined_test, subject_tom_only_weights)
        subject_tom_prediction_scores = helpers.get_prediction_scores(subject_tom_predictions, subject_test)
        tom_only_prediction_scores.append(subject_tom_prediction_scores)

        # Get and score predictions -- valence only weights
        subject_valence_only_weights = helpers.isolate_feature(subject_weights, start_index=3, end_index=4)
        subject_valence_predictions = helpers.get_predictions_from_weights(combined_test, subject_valence_only_weights)
        subject_valence_prediction_scores = helpers.get_prediction_scores(subject_valence_predictions, subject_test)
        valence_only_prediction_scores.append(subject_valence_prediction_scores)

        # Get and score predictions -- arousal only weights
        subject_arousal_only_weights = helpers.isolate_feature(subject_weights, start_index=4, end_index=5)
        subject_arousal_predictions = helpers.get_predictions_from_weights(combined_test, subject_arousal_only_weights)
        subject_arousal_prediction_scores = helpers.get_prediction_scores(subject_arousal_predictions, subject_test)
        arousal_only_prediction_scores.append(subject_arousal_prediction_scores)

#### Save results
np.save(os.path.join(directory, subject_group_str, 'encoding_results', (subject_group_str+'_weights_' + mask_str + '.npy')), weights)
print('Saved weights:', np.shape(weights))

np.save(os.path.join(directory, subject_group_str, 'encoding_results', 'unthresholded', (subject_group_str+'_full_scores_' + mask_str + '.npy')), full_prediction_scores)
print('Saved full prediction scores:', np.shape(full_prediction_scores))

np.save(os.path.join(directory, subject_group_str, 'encoding_results', 'unthresholded', (subject_group_str+'_moten_only_scores_' + mask_str + '.npy')), moten_only_prediction_scores)
print('Saved moten only prediction scores:', np.shape(moten_only_prediction_scores))

np.save(os.path.join(directory, subject_group_str, 'encoding_results', 'unthresholded', (subject_group_str+'_social_only_scores_' + mask_str + '.npy')), social_only_prediction_scores)
print('Saved social only prediction scores:', np.shape(social_only_prediction_scores))

np.save(os.path.join(directory, subject_group_str, 'encoding_results', 'unthresholded', (subject_group_str+'_faces_only_scores_' + mask_str + '.npy')), faces_only_prediction_scores)
print('Saved faces only prediction scores:', np.shape(faces_only_prediction_scores))

np.save(os.path.join(directory, subject_group_str, 'encoding_results', 'unthresholded', (subject_group_str+'_interaction_only_scores_' + mask_str + '.npy')), interaction_only_prediction_scores)
print('Saved interaction only prediction scores:', np.shape(interaction_only_prediction_scores))

np.save(os.path.join(directory, subject_group_str, 'encoding_results', 'unthresholded', (subject_group_str+'_tom_only_scores_' + mask_str + '.npy')), tom_only_prediction_scores)
print('Saved tom only prediction scores:', np.shape(tom_only_prediction_scores))

np.save(os.path.join(directory, subject_group_str, 'encoding_results', 'unthresholded', (subject_group_str+'_valence_only_scores_' + mask_str + '.npy')), valence_only_prediction_scores)
print('Saved valence only prediction scores:', np.shape(valence_only_prediction_scores))

np.save(os.path.join(directory, subject_group_str, 'encoding_results', 'unthresholded', (subject_group_str+'_arousal_only_scores_' + mask_str + '.npy')), arousal_only_prediction_scores)
print('Saved arousal only prediction scores:', np.shape(arousal_only_prediction_scores))
