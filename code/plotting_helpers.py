##### Imports

import matplotlib.pyplot as plt
import numpy as np
from nilearn import datasets, surface, plotting


##### Functions

def plot_surfaces(vol, title, vis_threshold):
    """
        Helper function to plot 4 views of a surface
    """

    # Generate surface
    fsaverage = datasets.fetch_surf_fsaverage()
    
    ## Generate surfaces
    right_surf = surface.vol_to_surf(vol, fsaverage.pial_right)
    left_surf = surface.vol_to_surf(vol, fsaverage.pial_left)
    
    ## Set up main figure
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title)
    
    ## Set up axes for each view 
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224, projection='3d')

     ## Left hemisphere
    ax1.title.set_text('Left hemisphere')
    
    # Lateral
    plotting.plot_surf_stat_map(fsaverage.infl_left, left_surf, 
                                hemi='left', view='lateral', bg_map=fsaverage.sulc_left,
                                colorbar=True, threshold=vis_threshold, 
                                axes=ax1)

    # Ventral
    plotting.plot_surf_stat_map(fsaverage.infl_left, left_surf, 
                                hemi='left', view='ventral', bg_map=fsaverage.sulc_left, 
                                colorbar=True, threshold=vis_threshold, 
                                axes=ax3)
    
    ## Right hemisphere
    ax2.title.set_text('Right hemisphere')
    
    # Lateral
    plotting.plot_surf_stat_map(fsaverage.infl_right, right_surf, 
                                hemi='right', view='lateral', bg_map=fsaverage.sulc_right,
                                colorbar=True, threshold=vis_threshold, 
                                axes=ax2)

    # Ventral
    plotting.plot_surf_stat_map(fsaverage.infl_right, right_surf, 
                                hemi='right', view='ventral', bg_map=fsaverage.sulc_right, 
                                colorbar=True, threshold=vis_threshold, 
                                axes=ax4)
    
    plotting.show()



def plot_lateral_only(vol, title, vis_threshold):
    """
        Helper function to plot 2 views of a surface 
    """

    # Generate surface
    fsaverage = datasets.fetch_surf_fsaverage()
    
    ## Generate surfaces
    right_surf = surface.vol_to_surf(vol, fsaverage.pial_right)
    left_surf = surface.vol_to_surf(vol, fsaverage.pial_left)
    
    ## Set up main figure
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title)
    
    ## Set up axes for each view 
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')

    ## Left hemisphere
    ax1.title.set_text('Left hemisphere')
    plotting.plot_surf_stat_map(fsaverage.infl_left, left_surf, 
                                hemi='left', view='lateral', bg_map=fsaverage.sulc_left,
                                colorbar=True, threshold=vis_threshold, 
                                axes=ax1)
    
    ## Right hemisphere
    ax2.title.set_text('Right hemisphere')
    plotting.plot_surf_stat_map(fsaverage.infl_right, right_surf, 
                                hemi='right', view='lateral', bg_map=fsaverage.sulc_right,
                                colorbar=True, threshold=vis_threshold, 
                                axes=ax2)
    
    plotting.show()

def plot_surfaces_with_vmax(vol, title, vis_threshold, vmax):
    """
        Helper function to plot 4 views of a surface with a fixed colorbar range
    """


    # Generate surface
    fsaverage = datasets.fetch_surf_fsaverage()

    ## Generate surfaces
    right_surf = surface.vol_to_surf(vol, fsaverage.pial_right)
    left_surf = surface.vol_to_surf(vol, fsaverage.pial_left)
    
    ## Set up main figure
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title)
    
    ## Set up axes for each view 
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224, projection='3d')

     ## Left hemisphere
    ax1.title.set_text('Left hemisphere')
    
    # Lateral
    plotting.plot_surf_stat_map(fsaverage.infl_left, left_surf, 
                                hemi='left', view='lateral', bg_map=fsaverage.sulc_left,
                                colorbar=True, threshold=vis_threshold, vmax=vmax,
                                axes=ax1)
  
    # Ventral
    plotting.plot_surf_stat_map(fsaverage.infl_left, left_surf, 
                                hemi='left', view='ventral', bg_map=fsaverage.sulc_left, 
                                colorbar=True, threshold=vis_threshold, vmax=vmax,
                                axes=ax3)
    
    ## Right hemisphere
    ax2.title.set_text('Right hemisphere')
    
    # Lateral
    plotting.plot_surf_stat_map(fsaverage.infl_right, right_surf, 
                                hemi='right', view='lateral', bg_map=fsaverage.sulc_right,
                                colorbar=True, threshold=vis_threshold, vmax=vmax,
                                axes=ax2)
   
    # Ventral
    plotting.plot_surf_stat_map(fsaverage.infl_right, right_surf, 
                                hemi='right', view='ventral', bg_map=fsaverage.sulc_right, 
                                colorbar=True, threshold=vis_threshold, vmax=vmax,
                                axes=ax4)
   
    
    plotting.show()

def plot_lateral_only_with_vmax(vol, title, vis_threshold, vmax):
    
    """
        Helper function to plot 2 views of a surface with a fixed colorbar range
    """    

    # Generate surface
    fsaverage = datasets.fetch_surf_fsaverage()
    
    ## Generate surfaces
    right_surf = surface.vol_to_surf(vol, fsaverage.pial_right)
    left_surf = surface.vol_to_surf(vol, fsaverage.pial_left)
    
    ## Set up main figure
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title)
    
    ## Set up axes for each view 
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')

    ## Left hemisphere
    ax1.title.set_text('Left hemisphere')
    plotting.plot_surf_stat_map(fsaverage.infl_left, left_surf, 
                                hemi='left', view='lateral', bg_map=fsaverage.sulc_left,
                                colorbar=True, threshold=vis_threshold, vmax=vmax,
                                axes=ax1)
    
    ## Right hemisphere
    ax2.title.set_text('Right hemisphere')
    plotting.plot_surf_stat_map(fsaverage.infl_right, right_surf, 
                                hemi='right', view='lateral', bg_map=fsaverage.sulc_right,
                                colorbar=True, threshold=vis_threshold, vmax=vmax,
                                axes=ax2)

