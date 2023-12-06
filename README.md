# Python and Bash code used for Analaysis
Summary: Raw code for banded ridge regression analysis of Partly Cloudy fMRI data

Hi! This is the code that was used in our paper "Early neural development of social perception: evidence from voxel-wise encoding in young children and adults"

To run this code, you will need to get the video stimulus from Richardson et al (2018), download tikreg and PyMoten packages from Gallant Lab Github (https://github.com/gallantlab).
To run ROI analyses, you would also need to download the ROI masks we used: We used STS mask from Ben Deen's group (https://bendeen.com/data/) and MT from Sabine Kastner's group (https://napl.scholar.princeton.edu/resources)

# How to use the code
Broadly, there's three types of files: 
1) Folder 'rockfish_scripts' has the codes for bash and python scripts used to run the encoding model, difference analysis between age groups, and thrsholding
2) Folder 'jupyter_notebooks' has the notebooks we used to visualize the data we analyzed using rockfish_scripts
3) Notebooks with 'sample' in its name gives you an example of how to use the notebooks

We also have helper files (ends with .py) that houses the functions we used to run the codes
