
##### Imports

# General package imports 
import os 
import numpy as np

# File loading packages
from scipy.io import loadmat
import pandas

# Brain-imaging related packages
import nibabel as nib
from nilearn import masking
from nilearn import image as niimage
from brainiak import image
from brainiak.isc import isc, permutation_isc

# Stats packages
from statsmodels.stats.multitest import multipletests
from sklearn import preprocessing 

###### Masking

def get_group_mask(subject_range, directory):

    """
        Returns a group intersect mask from a set of subject-specific masks for use in subsequent analysis.
        Note that this function assumes the format of mask filepaths.
    """
    masks = []

    for subject in subject_range:

        # Load subject specific mask
        mask_filename = os.path.join(directory, 'sub-pixar%03d/sub-pixar%03d_analysis_mask.nii.gz' % (subject, subject))
        mask = nib.load(mask_filename)
        masks.append(mask)

    group_mask = masking.intersect_masks(masks)

    return group_mask


def make_boolean_mask(mask):
    """
        Helper function to convert a NIFTI mask to a boolean mask 
    """
    return mask.get_fdata() > 0

def make_vol(voxels, mask):
    """
        Helper function to turn a 2D array of voxels into a volume using meta-data from a NIFTI mask
        returns NIFTI image of desired voxels.
    """

    # Make a blank volume of the correct size
    vol = np.zeros(np.shape(mask))

    # Identify the coordinates that are actually part of the brain
    mask_bool = make_boolean_mask(mask)
    coords = np.where(mask_bool)

    # Map the voxel array into brain space
    vol[coords] = voxels

    # Make a nii image of the isc map 
    nifti_vol = nib.Nifti1Image(vol, mask.affine, mask.header)

    return nifti_vol 

##### Loading and cleaning data
  
def clean_subject_data(subject, directory, save=False):

    """
        Applies subject specific mask and nuisance regressors to subject, then returns subject data.
        Optionally saves an nii file of the cleaned data to the original directory 
        Note that this function assumes the format of subject data filepath.
    """

    # Load functional images 
    func_filename = os.path.join(directory, 'sub-pixar%03d/sub-pixar%03d_task-pixar_run-001_swrf_bold.nii.gz' % (subject, subject))
    func_image = nib.load(func_filename)

    # Load subject specific mask
    mask_filename = os.path.join(directory, 'sub-pixar%03d/sub-pixar%03d_analysis_mask.nii.gz' % (subject, subject))
    mask = nib.load(mask_filename)
    
    # Load nuisance regressors
    nuisance_filename = os.path.join(directory, 'sub-pixar%03d/sub-pixar%03d_nuisance.mat' % (subject, subject))
    nuisance_regressors = loadmat(nuisance_filename)
    nuisance_regressors = nuisance_regressors.get('nuisance')
    
    # Attempt nilearn clean function 
    func_clean = niimage.clean_img(func_image,confounds=nuisance_regressors,t_r=2, mask_img=mask)

    # Save resulting image if save setting is enabled
    if save:
        save_filename = os.path.join(directory, 'sub-pixar%03d/sub-pixar%03d_clean.nii.gz' % (subject, subject))
        func_clean.to_filename(save_filename)

    return func_clean

def load_clean_data(subject_range, directory):
    """
        Loads previously cleaned data and returns it as an array.
        Note that this function assumes the format of subject data filepath.
    """
    clean_data = []

    for subject in subject_range:
        subject_filename = os.path.join(directory, 'sub-pixar%03d/sub-pixar%03d_clean.nii.gz' % (subject, subject))
        subject_data = nib.load(subject_filename)
        clean_data.append(subject_data)

    return clean_data


##### ISC mask generation

def perform_isc(bold_data, group_mask):
    """
        Formats and preps bold data and group mask, then runs brainiak's leave-one-out ISC function. 
        Returns ISC maps as an array.
    """

    # Convert group mask to boolean
    bool_mask =  make_boolean_mask(group_mask)

    # Mask and format bold_data to match brainiak ISC specifications 
    masked_images = image.mask_images(bold_data, bool_mask)
    formatted_images = image.MaskedMultiSubjectData.from_masked_images(masked_images, len(bold_data))

    # Perform leave-on-out ISC analysis
    isc_maps = isc(formatted_images, pairwise=False, tolerate_nans=True)

    return isc_maps

def cross_group_isc(bold_data, avg_adults, mask):
    """
    Takes in cleaned BOLD data for a subject group, then generates r-maps comparing each subject to an adult average.
    Returns a subjectxvoxels ISC map array.
    """
    isc_comparison = []
    for subject in bold_data:
        subject_isc = perform_isc(bold_data=np.array((subject, avg_adults)), group_mask=mask)
        isc_comparison.append(subject_isc)
    return isc_comparison


def threshold_isc(isc_maps,  threshold):
    """
        Runs a permutation test to control for multiple comparisons.
        Returns thresholded ISC values as an array, and the number of voxels that survived thresholding.
    """

    # Perform permutation test
    observed, p, distribution = permutation_isc(isc_maps, n_permutations=5000)

    # Removed nans from permutation test results

    # Create non-NaN mask
    nonnan_mask = ~np.isnan(observed)
    nonnan_coords = np.where(nonnan_mask)

    # Mask both the ISC and p-value map to exclude NaNs
    nonnan_isc = observed[nonnan_mask]
    nonnan_p = p[np.squeeze(nonnan_mask)]

    # Get FDR-controlled q-values
    nonnan_q = multipletests(nonnan_p, method='fdr_by')[1]

    num_significant_voxels = np.sum(nonnan_q < threshold)

    # Threshold ISCs according FDR-controlled threshold
    nonnan_isc[nonnan_q >= threshold] = np.nan

    # Reinsert thresholded ISCs back into whole brain image
    isc_thresholded = np.full(observed.shape, np.nan)
    isc_thresholded[nonnan_coords] = nonnan_isc

    # Swap from NaNs to zeros to make subsequent use as mask easier
    isc_thresholded = np.nan_to_num(isc_thresholded)

    return isc_thresholded, num_significant_voxels


def apply_isc_mask(bold_data, isc_thresholded, group_mask):
    """
        Makes a mask from thresholded ISC results and applies to bold data using metadata from a NIFTI group mask.
        Returns masked bold data.
    """

    # Make vol from thresholded ISC

    isc_nifti = make_vol(isc_thresholded, group_mask)
    isc_bool = make_boolean_mask(isc_nifti)

    # Apply mask to bold data 

    masked_data = []

    for subject in range(len(bold_data)):
        masked_image = image.mask_image(bold_data[subject], isc_bool)
        masked_data.append(masked_image)

    return masked_data

##### Trimming and normalization

def trim_blank_trs(bold_data):
    """
        Trims fMRI data from beginning and end of movie that should not be used.
        TRs 0-10 are blank, and TRs after 162 are credits, so these are considered the blank TRs.
        To account for hemodynamic lag, these blank TRs are offset by 2 TRs (4 seconds). 
        This yields a trimmed data set of TRs 12 - 164
        Returns an array of bold data with relevant slices removed
        
    """
    trimmed_data = []

    for data in bold_data: 
        # Remove TRs after 164
        end_trimmed = np.delete(data, slice(164,168), 1)
        
        # Remove TRS 0-12
        front_trimmed = np.delete(end_trimmed, slice(0,12), 1)
        
        trimmed_data.append(front_trimmed)
        
    return trimmed_data

def normalize_data(bold_data):
    """
        Uses sklearn.preprocessing.StandaradScalar() to normalize data for each subject.
        Returns an array of normalized bold data
    """
    normalized_data = []

    for data in bold_data: 
        bold_scaler = preprocessing.StandardScaler()
        bold_scaler.fit(data)
        normalized_bold = bold_scaler.transform(data)
        normalized_data.append(normalized_bold)
    
    return normalized_data


  
