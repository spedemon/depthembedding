# -*- coding: utf-8 -*-
# Harvard Medical School, Martinos Center for Biomedical Imaging 
# Aalto University, Department of Computer Science 

"""Calibrate cMiCE (Continuous Miniature Crystal Element, University of Washington) 
gamma detector with 8mm scintillation crystal, using the DepthEmbedding algorithm [1] 
and other methods for comparison (Centroid, 2D maximum-a-posteriori).
 
[1] 'Machine learning for the calibration of depth-of-interaction gamma-cameras.'
Stefano Pedemonte, Larry Pierce, Koen Van Leemput. 
Physics in Medicine and Biology, 2017."""

from depth_embedding import ReconstructorMAP, ReconstructorCentroid, get_cumulative_prior
from depth_embedding import BeerLambert, EnergySpectrum, HistogramCoordinates, \
    LikelihoodFilter
from depth_embedding import ModelInterpolator, Model2D, ModelDepthEmbedding, ModelMLEEM, \
    StatsCoordinates, model_error

import scipy
import scipy.io
from numpy import zeros, ones, sort, unravel_index, repeat, sum, where, squeeze, fliplr
from numpy import flipud, log, tile, float32, argsort, int32, histogram, linspace, round
from numpy import exp, convolve, sqrt, mgrid
import scipy.ndimage.filters
import numpy as np
import copy
import pylab as pl 


BATCH_SIZE = 128  # Number of events reconstructed in parallel (i.e. in a single batch). 
                  # If set to 0, all events are reconstructed at once: faster but may 
                  # starve memory when using large reconstruction grids 
                  # (e.g. when up-sampling the forward model). 


def get_data_cmice_grid(x,y,path="./20140508_ZA0082/test_data/"): 
    """Load cMiCe 8mm-crystal perpendicular photon beam experimental data. 
    The data is acquired using a photon beam perpendicular to the detector entrance 
    that scans the detector on a regular grid. This function loads the photon interaction 
    data (all photo-multiplier tubes readouts for each gamma photon interaction) for 
    a given position of the photon beam. 
    Data is stored in Matlab .mat files. Filenames are assumed to have this structure: 
    ROW%d_COL%d.mat, with the two numerical indexes running from 1 (Matlab indexing  
    convention) to Nx and from 1 to Ny respectively. This is how University of Washington 
    stores the cMiCe experimental data. 

    Args: 
        x (int): x position index of the photon beam. 
        y (int): y position index of the photon beam. 
        path (str): Location where the photon interaction data is stored. \
            Defaults to './20140508_ZA0082/test_data/'. 
            
    Returns: 
        ndarray: 2D photon interaction float matrix of sixe (N_events, N_detectors), \
            where N_events is the number of interaction events recorded 
            at the x,y grid location and N_detectors is the number of photomultiplier 
            tubes of cMiCe (i.e. 36). """
    filename = path + "/ROW%s_COL%s.mat"%(str(x+1).zfill(2), str(y+1).zfill(2))
    data = scipy.io.loadmat(filename)['data']
    data[data<0] = 0
    return float32(data.T)



class Cmice_8mm(): 
    """Calibration (i.e. estimation of the forward model) of the cMiCe PET camera using  
    various 2D and 3D methods. 
    This object provides all functions to load experimental photon beam data, 
    estimate the forward model using various algorithm, reconstruct 
    test photon beam interactions and evaluate reconstruction performance using 
    reconstruction methods and forward models: Centroid, 2D maximum-a-posteriori (MAP), 
    3D MAP with DepthEmbedding forward model, 3D MAP with DepthEmbedding+MLEEM forward 
    model. 
    
    Attributes:
        n_neighbours (int): number of neighbours used by DepthEmbedding. Default 12. 
        n_training (int): number of training iterations DepthEmbedding. Default 5000. 
        nz (int): number of depth bins DepthEmbedding. Default 16. 
        nx_resample (int): Number of pixels of final forward model in X. Default 32. 
        ny_resample (int): Number of pixels of final forward model in Y. Default 32. 
        nz_resample (int): Number of pixels of final forward model in Z. Default 16."""
    def __init__(self, n_neighbours=12, n_training=5000, nz=16, nx_resample=32, \
            ny_resample=32, nz_resample=16): 
        self.nx = 32
        self.ny = 32 
        self.nx_resample = nx_resample
        self.ny_resample = ny_resample
        self.nz_resample = nz_resample
        self.n_detectors = 64 
        self.scintillator_attenuation_coefficient = 0.83  #[cm-1]
        self.scintillator_thickness = 0.8                 #[cm]
        self.input_data_path = "./input_data_cmice_8mm/"
        self.results_path    = "./results_cmice_8mm/"
        self.nz = nz
        self.n_neighbours = n_neighbours
        self.n_training   = n_training 
        
        self._set_depth_prior()
        self.forward_model_2D = None 
        self.forward_model_DE = None 
        self.forward_model_MLEEM = None 
        self.coordinates_centroid = None 
        self.coordinates_2D_MAP = None 
        self.coordinates_3D_MAP_DE = None 
        self.coordinates_3D_MAP_MLEEM = None 
        self.coordinates_2D_MAP_filtered = None 
        self.coordinates_3D_MAP_DE_filtered = None 
        self.coordinates_3D_MAP_MLEEM_filtered = None
        self.histogram_centroid = None
        self.histogram_2D_MAP = None
        self.histogram_3D_MAP_DE = None
        self.histogram_3D_MAP_MLEEM = None
        self.spectrum_centroid = None
        self.spectrum_2D_MAP = None
        self.spectrum_3D_MAP_DE = None
        self.spectrum_3D_MAP_MLEEM = None
        self.spectrum_2D_MAP_filtered = None
        self.spectrum_3D_MAP_DE_filtered = None
        self.spectrum_3D_MAP_MLEEM_filtered = None
        self.likelihood_2D_MAP = None
        self.likelihood_3D_MAP_DE = None
        self.likelihood_3D_MAP_MLEEM = None
        self.likelihood_2D_MAP_filtered = None
        self.likelihood_3D_MAP_DE_filtered = None
        self.likelihood_3D_MAP_MLEEM_filtered = None


    def _set_depth_prior(self): 
        """Private function, load depth prior - Beer-Lambert function."""
        prior_model = BeerLambert(self.scintillator_attenuation_coefficient, self.nz, \
            self.scintillator_thickness/self.nz)
        self.prior = prior_model.get_probability_mass_function()

    def load_forward_model_2D(self, filename='/model_cmice_2D.mat'): 
        """Load 2D forward model from file. Returns the model as a 2D numpy array and 
        sets the model as an internal variable. 
        Args:
            filename (str): Matlab file containing 2D matrix of model parameters.
        Returns: 
            ndarray: model """
        filename = self.results_path+filename
        try: 
            model = scipy.io.loadmat(filename)['model_cmice_2D']
        except: 
            return None
        print "-Found 2D forward model file: %s - not recomputing it."%filename
        self.forward_model_2D = model
        return model

    def load_forward_model_DE(self, filename='/model_cmice_DE.mat'): 
        """Load DepthEmbedding forward model from file. Returns the model as a 3D numpy 
        array and sets the model as an internal variable. 
        Args:
            filename (str): Matlab file containing 3D matrix of model parameters.
        Returns: 
            ndarray: model """
        filename = self.results_path+filename
        try: 
            model = scipy.io.loadmat(filename)['model_cmice_DE']
        except: 
            return None
        print "-Found 3D DepthEmbedding forward model: %s - not recomputing it."%filename
        self.forward_model_DE = model
        return model

    def load_forward_model_MLEEM(self, filename='/model_cmice_MLEEM.mat'): 
        """Load DepthEmbedding-MLEEM forward model from file. Returns the model as a 3D 
        numpy array and sets the model as an internal variable. 
        Args:
            filename (str): Matlab file containing 3D matrix of model parameters.
        Returns: 
            ndarray: model """
        filename = self.results_path+filename
        try: 
            model = scipy.io.loadmat(filename)['model_cmice_MLEEM']
        except: 
            return None
        print "-Found 3D DepthEmbedding+MLEEM forward model: %s - no recompute."%filename
        self.forward_model_MLEEM = model
        return model


    def save_forward_model_2D(self, filename='/model_cmice_2D.mat'):
        """Saves 2D forward model to file as a Matlab matrix file. 
        Args:
            filename (str): name of file. """
        filename = self.results_path+filename 
        scipy.io.savemat(filename, {'model_cmice_2D':self.forward_model_2D})

    def save_forward_model_DE(self, filename='/model_cmice_DE.mat'):
        """Saves DepthEmbedding forward model to file as a Matlab matrix file. 
        Args:
            filename (str): name of file. """
        filename = self.results_path+filename 
        scipy.io.savemat(filename, {'model_cmice_DE':self.forward_model_DE})
 
    def save_forward_model_MLEEM(self, filename='/model_cmice_MLEEM.mat'): 
        """Saves DepthEmbedding-MLEEM forward model to file as a Matlab matrix file. 
        Args:
            filename (str): name of file. """
        filename = self.results_path+filename
        scipy.io.savemat(filename, {'model_cmice_MLEEM':self.forward_model_MLEEM})


    def load_reconstructions_grid_2D_MAP(self, filename='/coordinates_2D_MAP'): 
        """Load maximum-a-posteriori reconstruction grid 2D from Matlab file. 
        Returns the reconstructions and also saves them in internal variables. 
        Args:
            filename (str): base of file name: filename_nx_ny_nz.mat
        Returns:
            (coords, energy, likelihood) 
            Coordinates, energy and likelihood of each reconstructed event. 
            Each of these three variables is a list. 
            coords is a list of (x,y) tuples
            energy is a list of floats
            likelihood is a list of floats. """
        filename = self.results_path+filename
        filename = filename + '_%d_%d_%d.mat'%(self.nx_resample, self.ny_resample, \
            self.nz_resample)
        try: 
            coords = scipy.io.loadmat(filename)['coordinates_2D_MAP']
            energy = scipy.io.loadmat(filename)['energy_2D_MAP']
            likelihood = scipy.io.loadmat(filename)['likelihood_2D_MAP']
        except: 
            return None
        print "-Found 2D MAP reconstructed coordinates: %s - not recomputing."%filename
        for ix in range(len(coords)): 
            for iy in range(len(coords[0])):
                # for some reason, scipy.io.savemat adds an extra layer in the list, 
                # get read of it: 
                coords[ix][iy][0] = coords[ix][iy][0][0]
                coords[ix][iy][1] = coords[ix][iy][1][0]
                energy[ix][iy] = energy[ix][iy][0]
                likelihood[ix][iy] = likelihood[ix][iy][0]
        self.coordinates_2D_MAP = coords
        self.energy_2D_MAP = energy
        self.likelihood_2D_MAP = likelihood
        return coords, energy, likelihood

    def load_reconstructions_grid_3D_MAP_DE(self, filename='/coordinates_3D_MAP_DE'): 
        """Load DepthEmbedding maximum-a-posteriori reconstruction grid 3D from .mat file. 
        Returns the reconstructions and also saves them in internal variables. 
        Args:
            filename (str): base of file name: filename_nx_ny_nz.mat
        Returns:
            (coords, energy, likelihood) 
            Coordinates, energy and likelihood of each reconstructed event. 
            Each of these three variables is a list. 
            coords is a list of (x,y,z) tuples
            energy is a list of floats
            likelihood is a list of floats. """
        filename = self.results_path+filename
        filename = filename + '_%d_%d_%d.mat'%(self.nx_resample, self.ny_resample, \
            self.nz_resample)
        try: 
            coords = scipy.io.loadmat(filename)['coordinates_3D_MAP_DE']
            energy = scipy.io.loadmat(filename)['energy_3D_MAP_DE']
            likelihood = scipy.io.loadmat(filename)['likelihood_3D_MAP_DE']
        except: 
            return None
        print "-Found 3D MAP DepthEmbedding reconstructions: %s - no recompute."%filename
        for ix in range(len(coords)): 
            for iy in range(len(coords[0])):
                # scipy.io.savemat adds an extra layer in the list, remove it: 
                coords[ix][iy][0] = coords[ix][iy][0][0]
                coords[ix][iy][1] = coords[ix][iy][1][0]
                coords[ix][iy][2] = coords[ix][iy][2][0]
                energy[ix][iy] = energy[ix][iy][0]
                likelihood[ix][iy] = likelihood[ix][iy][0]
        self.coordinates_3D_MAP_DE = coords
        self.energy_3D_MAP_DE = energy
        self.likelihood_3D_MAP_DE = likelihood
        return coords, energy, likelihood

    def load_reconstructions_grid_3D_MAP_MLEEM(self,filename='/coordinates_3D_MAP_MLEEM'): 
        """Load DepthEmbedding+MLEEM max-a-post. reconstruction grid 3D from .mat file.  
        Returns the reconstructions and also saves them in internal variables. 
        Args:
            filename (str): base of file name: filename_nx_ny_nz.mat
        Returns:
            (coords, energy, likelihood) 
            Coordinates, energy and likelihood of each reconstructed event. 
            Each of these three variables is a list. 
            coords is a list of (x,y,z) tuples
            energy is a list of floats
            likelihood is a list of floats. """
        filename = self.results_path+filename
        filename = filename + '_%d_%d_%d.mat'%(self.nx_resample, self.ny_resample, \
            self.nz_resample)
        try: 
            coords = scipy.io.loadmat(filename)['coordinates_3D_MAP_MLEEM']
            energy = scipy.io.loadmat(filename)['energy_3D_MAP_MLEEM']
            likelihood = scipy.io.loadmat(filename)['likelihood_3D_MAP_MLEEM']
        except: 
            return None
        print "-Found 3D MAP DepthEmbedding+MLEEM reconstructions: %s - no recompute."\
            %filename
        for ix in range(len(coords)): 
            for iy in range(len(coords[0])):
                # scipy.io.savemat adds an extra layer in the list, remove it: 
                coords[ix][iy][0] = coords[ix][iy][0][0]
                coords[ix][iy][1] = coords[ix][iy][1][0]
                coords[ix][iy][2] = coords[ix][iy][2][0]
                energy[ix][iy] = energy[ix][iy][0]
                likelihood[ix][iy] = likelihood[ix][iy][0]
        self.coordinates_3D_MAP_MLEEM = coords
        self.energy_3D_MAP_MLEEM = energy
        self.likelihood_3D_MAP_MLEEM = likelihood
        return coords, energy, likelihood


    def save_reconstructions_grid_2D_MAP(self, filename='/coordinates_2D_MAP'): 
        """Save maximum-a-posteriori reconstruction grid 2D to .mat file.  
        Returns the reconstructions and also saves them in internal variables. 
        Args:
            filename (str): base of file name: filename_nx_ny_nz.mat"""
        filename = self.results_path+filename
        filename = filename + '_%d_%d_%d.mat'%(self.nx_resample, self.ny_resample, \
            self.nz_resample)
        scipy.io.savemat(filename, {'coordinates_2D_MAP':self.coordinates_2D_MAP,\
            'energy_2D_MAP':self.energy_2D_MAP,\
            'likelihood_2D_MAP':self.likelihood_2D_MAP})

    def save_reconstructions_grid_3D_MAP_DE(self, filename='/coordinates_3D_MAP_DE'): 
        """Save DepthEmbedding maximum-a-posteriori reconstruction grid 3D to .mat file.  
        Returns the reconstructions and also saves them in internal variables. 
        Args:
            filename (str): base of file name: filename_nx_ny_nz.mat"""
        filename = self.results_path+filename
        filename = filename + '_%d_%d_%d.mat'%(self.nx_resample, self.ny_resample, \
            self.nz_resample)
        scipy.io.savemat(filename, {'coordinates_3D_MAP_DE':self.coordinates_3D_MAP_DE,\
            'energy_3D_MAP_DE':self.energy_3D_MAP_DE,\
            'likelihood_3D_MAP_DE':self.likelihood_3D_MAP_DE})

    def save_reconstructions_grid_3D_MAP_MLEEM(self,filename='/coordinates_3D_MAP_MLEEM'):
        """Save DepthEmbedding+MLEEM max.-a-post. reconstruction grid 3D to .mat file.  
        Returns the reconstructions and also saves them in internal variables. 
        Args:
            filename (str): base of file name: filename_nx_ny_nz.mat"""
        filename = self.results_path+filename
        filename = filename + '_%d_%d_%d.mat'%(self.nx_resample, self.ny_resample, \
            self.nz_resample)
        scipy.io.savemat(filename, {\
            'coordinates_3D_MAP_MLEEM':self.coordinates_3D_MAP_MLEEM,\
            'energy_3D_MAP_MLEEM':self.energy_3D_MAP_MLEEM,\
            'likelihood_3D_MAP_MLEEM':self.likelihood_3D_MAP_MLEEM})



    def estimate_forward_model_2D(self): 
        """Estimate 2D formard model from photon interacrion measurements. The function 
        returns the forward model and the model is stored in an internal variable - the 
        object maintains the state in order to provide functions to save the model, 
        plot information, compute metrics, and reconstruct events of interaction using the 
        forward model. 
        Returns:
            ndarray: 2D matrix (self.nx, self.ny) parameters of forward model."""
        model = zeros([self.nx, self.ny, self.n_detectors])
        for ix in range(self.nx): 
            self._print_percentage(ix,self.nx)
            for iy in range(self.ny): 
                calibration_data = self.get_data_grid(ix,iy)
                model_estimator = Model2D()
                model_estimator.set_calibration_data(calibration_data)
                model[ix,iy,:] = model_estimator.estimate_forward_model()
        self.forward_model_2D = model 
        return self.forward_model_2D

    def estimate_forward_model_DE(self): 
        """Estimate 3D formard model from photon interacrion measurements using 
        DepthEmbedding. 
        The function returns the forward model and the model is stored in an internal 
        variable - the object maintains the state in order to provide functions to save 
        the model, plot information, compute metrics, and reconstruct events of 
        interaction using the forward model. 
        Returns: 
            ndarray: 3D matrix (self.nx, self.ny, self.nz) parameters of forward model."""
        model = zeros([self.nx, self.ny, self.nz, self.n_detectors])
        for ix in range(self.nx): 
            self._print_percentage(ix,self.nx)
            for iy in range(self.ny): 
                calibration_data = self.get_data_grid(ix,iy)
                filter = LikelihoodFilter(self.forward_model_2D)
                calibration_data_filtered, rejected = filter.filter(calibration_data, \
                    ix,iy, points = self.n_training) 
                print "Number of points for location %d,%d:"%(ix,iy), \
                    calibration_data_filtered.shape[0]
                model_estimator = ModelDepthEmbedding(nz=self.nz, \
                    n_neighbours=self.n_neighbours)
                model_estimator.set_calibration_data(calibration_data_filtered)
                model_estimator.set_depth_prior(self.prior) 
                m = model_estimator.estimate_forward_model(unit_norm=False, \
                    zero_mean=False) 
                print "Max expected signal:    z=0: %2.2f    z=%d: %2.2f"%(m[0,:].max(), \
                    self.nz-1, m[self.nz-1,:].max())
                model[ix,iy,:,:] = m 
        self.forward_model_DE = model 
        return self.forward_model_DE
    
    def estimate_forward_model_MLEEM(self): 
        """Estimate 3D formard model from photon interacrion measurements using 
        DepthEmbedding + MLEEM refinement. 
        The function returns the forward model and the model is stored in an internal 
        variable - the object maintains the state in order to provide functions to save 
        the model, plot information, compute metrics, and reconstruct events of 
        interaction using the forward model. 
        Returns: 
            ndarray: 3D matrix (self.nx, self.ny, self.nz) parameters of forward model."""
        model = zeros([self.nx, self.ny, self.nz, self.n_detectors])
        for ix in range(self.nx): 
            self._print_percentage(ix,self.nx)
            for iy in range(self.ny): 
                calibration_data = self.get_data_grid(ix,iy)
                filter = LikelihoodFilter(self.forward_model_2D)
                calibration_data_filtered, rejected = filter.filter(calibration_data, \
                    ix,iy, points = self.n_training) 
                model_estimator=ModelMLEEM(initial_model=self.forward_model_DE[ix,iy,:,:])
                model_estimator.set_calibration_data(calibration_data_filtered) 
                model_estimator.set_depth_prior(self.prior) 
                model[ix,iy,:,:] = model_estimator.estimate_forward_model( \
                    n_max_iterations=200, method='soft', smoothness=0.0, prune=False) 
        self.forward_model_MLEEM = model 
        return self.forward_model_MLEEM


    def resample_forward_model_2D(self, grid_size, interpolation='cubic',noise_sigma=0.5):
        """Resample (re-intrepolate) the 2D forward model.
        The function resamples the forward model stored internally in this object. 
        It returns the resampled forward model and replaces the stored forward model with 
        the new, resampled, model. 
        Args: 
            grid_size ((int,int,int)): number of pixels of resampled grid (nx,ny,1)
            The third value (nz) is ignored. 
        Returns: 
            ndarray: 2D matrix (self.nx,self.ny) parameters of resampled forward model."""
        if self.nx == self.nx_resample and self.ny == self.ny_resample and \
            self.nz == self.nz_resample: 
            print "Not resampling 2D model."
            return self.forward_model_2D
        interpolator = ModelInterpolator()
        forward_model = interpolator.resample(self.forward_model_2D, grid_size, \
            interpolation=interpolation)
        if noise_sigma > 1e-9:
            # A small amount of noise in the forward model removes interpolation artefacts 
            forward_model = np.random.normal(forward_model, noise_sigma)
            forward_model[forward_model<=0.0]=0.0
        self.forward_model_2D = forward_model
        return forward_model

    def resample_forward_model_DE(self, grid_size, interpolation='cubic',noise_sigma=0.5):
        """Resample (re-intrepolate) the 3D DepthEmbedding forward model.
        The function resamples the forward model stored internally in this object. 
        It returns the resampled forward model and replaces the stored forward model with 
        the new, resampled, model. 
        Args: 
            grid_size ((int,int,int)): number of pixels of resampled grid (nx,ny,1)
        Returns: 
            ndarray: 3D matrix (self.nx,self.ny) parameters of resampled forward model."""
        if self.nx == self.nx_resample and self.ny == self.ny_resample and \
            self.nz == self.nz_resample: 
            print "Not resampling DE model."
            return self.forward_model_DE
        interpolator = ModelInterpolator()
        forward_model = interpolator.resample(self.forward_model_DE, grid_size, \
            interpolation=interpolation)
        if noise_sigma > 1e-9:
            # A small amount of noise in the forward model removes interpolation artefacts 
            forward_model = np.random.normal(forward_model, noise_sigma)
            forward_model[forward_model<=0.0]=0.0
        self.forward_model_DE = forward_model
        return forward_model

    def resample_forward_model_MLEEM(self,grid_size,interpolation='cubic',noise_sigma=.5):
        """Resample (re-intrepolate) the 3D DepthEmbedding+MLEEM forward model.
        The function resamples the forward model stored internally in this object. 
        It returns the resampled forward model and replaces the stored forward model with 
        the new, resampled, model. 
        Args: 
            grid_size ((int,int,int)): number of pixels of resampled grid (nx,ny,1)
        Returns: 
            ndarray: 3D matrix (self.nx,self.ny) parameters of resampled forward model."""
        if self.nx == self.nx_resample and self.ny == self.ny_resample and \
            self.nz == self.nz_resample: 
            print "Not resampling MLEEM model."
            return self.forward_model_MLEEM
        interpolator = ModelInterpolator()
        forward_model = interpolator.resample(self.forward_model_MLEEM, grid_size, \
            interpolation=interpolation)
        if noise_sigma > 1e-9:
            # A small amount of noise in the forward model removes interpolation artefacts 
            forward_model = np.random.normal(forward_model, noise_sigma)
            forward_model[forward_model<=0.0]=0.0
        self.forward_model_MLEEM = forward_model
        return forward_model


    def load_test_grid(self, x_locations, y_locations): 
        """Loads photon beam grid to evaluate reconstruction performance using the loaded 
        forward models. 
        The function returns the grid as a matrix and the grid matrix is stored internally
        in the object: self.grid.
        After calling this function, call self.reconstruct_grid_xxx functions to 
        reconstruct events of interaction at the grid locations specified here. 
        Then use self.make_histogram_xxx to create photon counting histograms. 
        Args: 
            x_locations (list of int): x locations of photon beams, unit measure is pixels
            y_locations (list of int): y locations of photon beams, unit measure is pixels
            e.g. load_test_grid([1,6,11,16,21,26,31], [1,6,11,16,21,26,31])
        Returns: 
            ndarray: 2D matrix (len(x_locations),len(x_locations)) of grid locations."""
        self.grid = []
        for ix in x_locations: 
            grid_row = [] 
            for iy in y_locations: 
                data = self.get_data_grid(ix,iy)
                grid_row.append(data)
            self.grid.append(grid_row) 
        self.grid_shape = (len(x_locations),len(y_locations))
        tiles =tile(int32(y_locations).reshape((len(y_locations),1)),(1,len(x_locations)))
        tiles = tiles.transpose()
        self.grid_locations = [tile(x_locations,(len(y_locations),1)).transpose(), tiles]
        return self.grid


            
    def reconstruct_grid_centroid(self, shiftx=2.9, shifty=2.5, scale=8.0): 
        """Reconstruct photon interactions on test grid using basic Centroid algorithm 
        (2D). 
        This function produces a list of coordinates of interaction estimates for the 
        interaction events. The list is also stored in an internal object variable: 
        self.coordinates_centroid. After calling this function, 
        call make_histograms_grid_centroid() to copute 
        histograms of the photon counts on the reconstruction grid. 
        
        Args:
            shiftx (float): shift parameter of centroid algorithm along x axis \
                - determines locations of reconstructed grid
            shiftx (float): shift parameter of centroid algorithm along x axis \
                - determines locations of reconstructed grid
            scale (float): scale parameter of centroid algorithm - determines scane of \
                reconstructed grid. 

        Returns:
            ndarray: coordinates of interaction. """
        self.coordinates_centroid = [] 
        self.energy_centroid = []
        x_detectors = tile(linspace(0,9,8),(1,8))[0] - 4.5 
        y_detectors = repeat(linspace(0,9,8),8,axis=0) - 4.3
        reconstructor = ReconstructorCentroid(x_detectors=x_detectors, \
        y_detectors=y_detectors, x_max=self.nx_resample-1, y_max=self.ny_resample-1, \
        shiftx=shiftx, shifty=shifty, scale=scale, exponent=1.0)  
        for ix in range(self.grid_shape[0]): 
            row_coordinates = [] 
            row_energy = []
            for iy in range(self.grid_shape[1]):
                data = self.grid[ix][iy]
                [xrec,yrec,energyrec] = reconstructor.reconstruct(data) 
                row_coordinates.append([xrec,yrec])
                row_energy.append(energyrec)
            self.coordinates_centroid.append(row_coordinates)
            self.energy_centroid.append(row_energy) 
        return self.coordinates_centroid

    def make_histograms_grid_centroid(self): 
        """Compute histograms (photon counts) at reconstruction grid locations for 
        reconstructions obtained using the 2D Centroid algorithm. 
        The histogram is stored in self.histogram_centroid. 
        The function also computes the energy spectrum of the reconstructed events and 
        stores it in self.spectrum_centroid. 
        """
        self.histogram_centroid = HistogramCoordinates(self.nx_resample, self.ny_resample)
        self.spectrum_centroid = EnergySpectrum(scale="auto", peak=511.0, max_energy=1500) 
        for ix in range(self.grid_shape[0]): 
            for iy in range(self.grid_shape[1]):
                [xrec,yrec] = self.coordinates_centroid[ix][iy]
                energyrec = self.energy_centroid[ix][iy]
                self.histogram_centroid.add_data([xrec,yrec])
                self.spectrum_centroid.add_data(energyrec)

    def reconstruct_grid_2D_MAP(self): 
        """Reconstruct photon interactions on test grid using 2D maximum a posteriori
        algorithm. 
        This function produces a list of coordinates of interaction estimates for the 
        interaction events. The list is also stored in an internal object variable: 
        self.coordinates_2D_MAP. After calling this function, 
        call make_histograms_grid_2D_MAP() to copute 
        histograms of the photon counts on the reconstruction grid. 

        Returns:
            ndarray: coordinates of interaction. """
        self.coordinates_2D_MAP = []
        self.energy_2D_MAP = []
        self.likelihood_2D_MAP = []
        reconstructor = ReconstructorMAP(self.forward_model_2D)  
        for ix in range(self.grid_shape[0]): 
            #self._print_percentage(ix,self.grid_shape[0])
            row_coordinates = []
            row_energy = []
            row_likelihood = []
            for iy in range(self.grid_shape[1]):
                N = self.grid_shape[0]*self.grid_shape[1]
                self._print_percentage(iy+ix*self.grid_shape[1],N)
                data = self.grid[ix][iy]
                [xrec,yrec,energyrec,logposterior] = reconstructor.reconstruct(data, \
                    batch_size=BATCH_SIZE) 
                row_coordinates.append([xrec,yrec])
                row_energy.append(energyrec)
                row_likelihood.append(logposterior)
            self.coordinates_2D_MAP.append(row_coordinates)
            self.energy_2D_MAP.append(row_energy)
            self.likelihood_2D_MAP.append(row_likelihood)
        return self.coordinates_2D_MAP
    
    def make_histograms_grid_2D_MAP(self): 
        """Compute histograms (photon counts) at reconstruction grid locations for 
        reconstructions obtained using the 2D maximum-a-posteriori. 
        The histogram is stored in self.histogram_2D_MAP. 
        The function also computes the energy spectrum of the reconstructed events and 
        stores it in self.spectrum_2D_MAP. 
        """
        self.histogram_2D_MAP = HistogramCoordinates(self.nx_resample, self.ny_resample)
        self.spectrum_2D_MAP = EnergySpectrum(scale="auto", peak=511.0, max_energy=1500) 
        for ix in range(self.grid_shape[0]): 
            for iy in range(self.grid_shape[1]):
                [xrec,yrec] = self.coordinates_2D_MAP_filtered[ix][iy]
                energyrec = self.energy_2D_MAP_filtered[ix][iy]
                self.histogram_2D_MAP.add_data([xrec,yrec])
                self.spectrum_2D_MAP.add_data(energyrec)

    def reconstruct_grid_3D_MAP_DE(self): 
        """Reconstruct photon interactions on test grid using 3D maximum a posteriori
        algorithm and DepthEmbedding forward model. 
        This function produces a list of coordinates of interaction estimates for the 
        interaction events. The list is also stored in an internal object variable: 
        self.coordinates_2D_MAP_DE. After calling this function, 
        call make_histograms_grid_3D_MAP_DE() to copute 
        histograms of the photon counts on the reconstruction grid. 

        Returns:
            ndarray: coordinates of interaction. """
        self.coordinates_3D_MAP_DE = []
        self.energy_3D_MAP_DE = []
        self.likelihood_3D_MAP_DE = []
        reconstructor = ReconstructorMAP(self.forward_model_DE)  
        reconstructor.set_prior(self.prior)
        for ix in range(self.grid_shape[0]): 
            #self._print_percentage(ix,self.grid_shape[0])
            row_coordinates = []
            row_energy = []
            row_likelihood = [] 
            for iy in range(self.grid_shape[1]):
                N = self.grid_shape[0]*self.grid_shape[1]
                self._print_percentage(iy+ix*self.grid_shape[1],N)
                data = self.grid[ix][iy]
                [xrec,yrec,zrec,energyrec,logposterior] = reconstructor.reconstruct(data,\
                    batch_size=BATCH_SIZE) 
                row_coordinates.append([xrec,yrec,zrec])
                row_energy.append(energyrec)
                row_likelihood.append(logposterior)
            self.coordinates_3D_MAP_DE.append(row_coordinates)
            self.energy_3D_MAP_DE.append(row_energy)
            self.likelihood_3D_MAP_DE.append(row_likelihood)
        return self.coordinates_3D_MAP_DE

    def make_histograms_grid_3D_MAP_DE(self):
        """Compute histograms (photon counts) at reconstruction grid locations for 
        reconstructions obtained using the 3D maximum-a-posteriori with DepthEmbedding 
        forward model. 
        The histogram is stored in self.histogram_3D_MAP_DE. 
        The function also computes the energy spectrum of the reconstructed events and 
        stores it in self.spectrum_3D_MAP_DE. 
        """
        self.histogram_3D_MAP_DE =HistogramCoordinates(self.nx_resample,self.ny_resample,\
             self.nz_resample)
        self.spectrum_3D_MAP_DE =EnergySpectrum(scale="auto", peak=511.0, max_energy=1500) 
        for ix in range(self.grid_shape[0]): 
            for iy in range(self.grid_shape[1]):
                [xrec,yrec,zrec] = self.coordinates_3D_MAP_DE_filtered[ix][iy]
                energyrec = self.energy_3D_MAP_DE_filtered[ix][iy]
                self.histogram_3D_MAP_DE.add_data([xrec,yrec,zrec])
                #print ix,iy,energyrec
                self.spectrum_3D_MAP_DE.add_data(energyrec)

    def reconstruct_grid_3D_MAP_MLEEM(self): 
        """Reconstruct photon interactions on test grid using 3D maximum a posteriori
        algorithm and DepthEmbedding+MLEEM forward model. 
        This function produces a list of coordinates of interaction estimates for the 
        interaction events. The list is also stored in an internal object variable: 
        self.coordinates_2D_MAP_DE_MLEEM. After calling this function, 
        call make_histograms_grid_3D_MAP_DE_MLEEM() to copute 
        histograms of the photon counts on the reconstruction grid. 

        Returns:
            ndarray: coordinates of interaction. """
        self.coordinates_3D_MAP_MLEEM = []
        self.energy_3D_MAP_MLEEM = []
        self.likelihood_3D_MAP_MLEEM = []
        reconstructor = ReconstructorMAP(self.forward_model_MLEEM)  
        reconstructor.set_prior(self.prior)
        for ix in range(self.grid_shape[0]): 
            #self._print_percentage(ix,self.grid_shape[0])
            row_coordinates = []
            row_energy = []
            row_likelihood = []
            for iy in range(self.grid_shape[1]):
                N = self.grid_shape[0]*self.grid_shape[1]
                self._print_percentage(iy+ix*self.grid_shape[1],N)
                data = self.grid[ix][iy]
                [xrec,yrec,zrec,energyrec,logposterior] = reconstructor.reconstruct(data,\
                    batch_size=BATCH_SIZE) 
                row_coordinates.append([xrec,yrec,zrec])
                row_energy.append(energyrec)
                row_likelihood.append(logposterior)
            self.coordinates_3D_MAP_MLEEM.append(row_coordinates)
            self.energy_3D_MAP_MLEEM.append(row_energy)
            self.likelihood_3D_MAP_MLEEM.append(row_likelihood)
        return self.coordinates_3D_MAP_MLEEM

    def make_histograms_grid_3D_MAP_MLEEM(self):
        """Compute histograms (photon counts) at reconstruction grid locations for 
        reconstructions obtained using the 3D maximum-a-posteriori with 
        DepthEmbedding+MLEEM forward model. 
        The histogram is stored in self.histogram_3D_MAP_DE_MLEEM. 
        The function also computes the energy spectrum of the reconstructed events and 
        stores it in self.spectrum_3D_MAP_DE_MLEEM. 
        """
        self.histogram_3D_MAP_MLEEM = HistogramCoordinates(self.nx_resample, \
            self.ny_resample, self.nz_resample)
        self.spectrum_3D_MAP_MLEEM=EnergySpectrum(scale="auto",peak=511.0,max_energy=1500) 
        for ix in range(self.grid_shape[0]): 
            for iy in range(self.grid_shape[1]):
                [xrec,yrec,zrec] = self.coordinates_3D_MAP_MLEEM_filtered[ix][iy]
                energyrec = self.energy_3D_MAP_MLEEM_filtered[ix][iy]
                self.histogram_3D_MAP_MLEEM.add_data([xrec,yrec,zrec])
                self.spectrum_3D_MAP_MLEEM.add_data(energyrec)


    def filter_events_2D_MAP(self, fraction=0.5): 
        """Filter events reconstructed using 2D maximum-a-posteriori based on 
        maximum-probability value. Use 'fraction' fraction of events with highest values 
        of maximum-probability, dicard the remaining 1-'fraction' events. 
        The coordinates, energy and maximum-prob of the filtered events are stored in: 
        self.coordinates_2D_MAP_filtered, 
        self.energy_2D_MAP_filtered,
        self.likelihood_2D_MAP_filtered. 
        
        Args: 
            fraction (float): fraction of events not discarded. \
                0: discard all, 1: keep all."""
        self.coordinates_2D_MAP_filtered = []
        self.energy_2D_MAP_filtered = [] 
        self.likelihood_2D_MAP_filtered = []
        for ix in range(self.grid_shape[0]): 
            C = []
            E = []
            L = []
            for iy in range(self.grid_shape[1]):
                N = int32(len(self.coordinates_2D_MAP[ix][iy][0])*fraction)
                x = self.coordinates_2D_MAP[ix][iy][0]
                y = self.coordinates_2D_MAP[ix][iy][1]
                e = self.energy_2D_MAP[ix][iy]
                l = self.likelihood_2D_MAP[ix][iy]
                indexes = flipud(argsort(l)) 
                x = x[indexes][0:N]
                y = y[indexes][0:N]
                e = e[indexes][0:N]
                l = l[indexes][0:N]
                C.append([x,y])
                E.append(e)
                L.append(l)
            self.coordinates_2D_MAP_filtered.append(C) 
            self.energy_2D_MAP_filtered.append(E) 
            self.likelihood_2D_MAP_filtered.append(L)

    def filter_events_3D_MAP_DE(self, fraction=0.5): 
        """Filter events reconstructed using 3D maximum-a-posteriori with DepthEmbedding 
        forward model, based on maximum-probability value. 
        Use 'fraction' fraction of events with highest values 
        of maximum-probability, dicard the remaining 1-'fraction' events. 
        The coordinates, energy and maximum-prob of the filtered events are stored in: 
        self.coordinates_3D_MAP_DE_filtered, 
        self.energy_3D_MAP_DE_filtered,
        self.likelihood_3D_MAP_DE_filtered. 
        
        Args: 
            fraction (float): fraction of events not discarded. \
                0: discard all, 1: keep all."""
        self.coordinates_3D_MAP_DE_filtered = []
        self.energy_3D_MAP_DE_filtered = []
        self.likelihood_3D_MAP_DE_filtered = []
        for ix in range(self.grid_shape[0]): 
            C = []
            E = []
            L = []
            for iy in range(self.grid_shape[1]):
                N = int32(len(self.coordinates_3D_MAP_DE[ix][iy][0])*fraction)
                x = self.coordinates_3D_MAP_DE[ix][iy][0]
                y = self.coordinates_3D_MAP_DE[ix][iy][1]
                z = self.coordinates_3D_MAP_DE[ix][iy][2]
                e = self.energy_3D_MAP_DE[ix][iy]
                l = self.likelihood_3D_MAP_DE[ix][iy]
                indexes = flipud(argsort(l)) 
                x = x[indexes][0:N]
                y = y[indexes][0:N]
                z = z[indexes][0:N]
                e = e[indexes][0:N]
                l = l[indexes][0:N]
                C.append([x,y,z])
                E.append(e)
                L.append(l)
            self.coordinates_3D_MAP_DE_filtered.append(C) 
            self.energy_3D_MAP_DE_filtered.append(E) 
            self.likelihood_3D_MAP_DE_filtered.append(L)
        
    def filter_events_3D_MAP_MLEEM(self, fraction=0.5): 
        """Filter events reconstructed using 3D maximum-a-posteriori with 
        DepthEmbedding+MLEEM forward model, based on maximum-probability value. 
        Use 'fraction' fraction of events with highest values 
        of maximum-probability, dicard the remaining 1-'fraction' events. 
        The coordinates, energy and maximum-prob of the filtered events are stored in: 
        self.coordinates_3D_MAP_MLEEM_filtered, 
        self.energy_3D_MAP_MLEEM_filtered,
        self.likelihood_3D_MAP_MLEEM_filtered. 
        
        Args: 
            fraction (float): fraction of events not discarded. \
                0: discard all, 1: keep all."""
        self.coordinates_3D_MAP_MLEEM_filtered = []
        self.energy_3D_MAP_MLEEM_filtered = []
        self.likelihood_3D_MAP_MLEEM_filtered = []
        for ix in range(self.grid_shape[0]): 
            C = []
            E = []
            L = []
            for iy in range(self.grid_shape[1]):
                N = int32(len(self.coordinates_3D_MAP_MLEEM[ix][iy][0])*fraction)
                x = self.coordinates_3D_MAP_MLEEM[ix][iy][0]
                y = self.coordinates_3D_MAP_MLEEM[ix][iy][1]
                z = self.coordinates_3D_MAP_MLEEM[ix][iy][2]
                e = self.energy_3D_MAP_MLEEM[ix][iy]
                l = self.likelihood_3D_MAP_MLEEM[ix][iy]
                indexes = flipud(argsort(l)) 
                x = x[indexes][0:N]
                y = y[indexes][0:N]
                z = z[indexes][0:N]
                e = e[indexes][0:N]
                l = l[indexes][0:N]
                C.append([x,y,z])
                E.append(e)
                L.append(l)
            self.coordinates_3D_MAP_MLEEM_filtered.append(C) 
            self.energy_3D_MAP_MLEEM_filtered.append(E) 
            self.likelihood_3D_MAP_MLEEM_filtered.append(L)

    def _scale_grid_locations(self): 
        """Scale locations of test grid. If the forward models have been resampled, 
        it is necessary to call this function in order to compute metrics of the 
        reconstruction algorithms using the photon beams centered on the test grid. 
        This function omputes the new test grid locations on the 
        resampled forward model grid. 
        This is used in function self.compute_bias_and_variance_grid()
        
        Returns: 
            ndarray: 2D matrix of scaled test grid locations."""
        n_points_x = self.grid_locations[0].shape[0]
        n_points_y = self.grid_locations[0].shape[1]
        grid_locations_scaled = copy.deepcopy(self.grid_locations)
        for ix in range(n_points_x): 
            for iy in range(n_points_y): 
                grid_locations_scaled[0][ix][iy] = grid_locations_scaled[0][ix][iy] * \
                    1.0 / self.nx * self.nx_resample
                grid_locations_scaled[1][ix][iy] = grid_locations_scaled[1][ix][iy] * \
                    1.0 / self.ny * self.ny_resample
        return grid_locations_scaled

    def compute_bias_and_variance_grid(self): 
        """Compute bias and variance of photon beam reconstructions using all 
        reconstruction algorithms: Centroid, 2D MAP, 3D MAP DepthEmbedding, 
        3D MAP DepthEmbedding+MLEEM. 
        The bias and variance are printed to stdout. """
        grid_locations = self._scale_grid_locations() 
        if self.coordinates_centroid is not None: 
            stats_centroid = StatsCoordinates(self.coordinates_centroid, grid_locations, \
                [(0,self.nx_resample),(0,self.ny_resample)], "Centroid", \
                scale=(50.0/self.nx_resample,50.0/self.ny_resample,8.0/self.nz_resample))
            stats_centroid.print_summary()
        if self.coordinates_2D_MAP_filtered is not None: 
            stats_2D_MAP = StatsCoordinates(self.coordinates_2D_MAP_filtered, \
                grid_locations, [(0,self.nx_resample),(0,self.ny_resample)], "2D MAP", \
                scale=(50.0/self.nx_resample,50.0/self.ny_resample,8.0/self.nz_resample))
            stats_2D_MAP.print_summary()
        if self.coordinates_3D_MAP_DE_filtered is not None: 
            stats_3D_MAP_DE = StatsCoordinates(self.coordinates_3D_MAP_DE_filtered, \
                grid_locations, \
                [(0,self.nx_resample),(0,self.ny_resample),(2,self.nz_resample-2)], \
                "3D MAP DepthEmbedding", \
                scale=(50.0/self.nx_resample,50.0/self.ny_resample,8.0/self.nz_resample))
            stats_3D_MAP_DE.print_summary()
            #stats_3D_MAP_DE.plot_depth_density(6)
        if self.coordinates_3D_MAP_MLEEM_filtered is not None: 
            stats_3D_MAP_MLEEM = StatsCoordinates(self.coordinates_3D_MAP_MLEEM_filtered,\
                grid_locations, \
                [(0,self.nx_resample),(0,self.ny_resample),(2,self.nz_resample-2)], \
                "3D MAP DepthEmbedding+MLEEM", \
                scale=(50.0/self.nx_resample,50.0/self.ny_resample,8.0/self.nz_resample))
            stats_3D_MAP_MLEEM.print_summary()
            #stats_3D_MAP_MLEEM.plot_depth_density(6)
            

    def visualize_results_grid(self, n_points=4000, ix=15, iy=15, vmax=1000): 
        """Display images that provide insight on the reconstruction of photon beam 
        interactions on grid. Where applicable, the images are created for each 
        reconstruction algorithm: Centroid, 2D MAP, 3D MAP DepthEmbedding, 
        3D MAP DepthEmbedding+MLEEM. 
        The images include: 
        - 3D DepthEmbedding manifold for interactions at a given grid location,
        - 2D Histogram of estimated locations of interaction for all grid locations, 
        - Energy histogram for all interaction events, 
        - Distribution of reconstructed depth-of-interaction (vs Beer-Lambert model).
        
        Args: 
            n_points (int): n_points utilized to plot DepthEmbedding manifold. 
            ix (int): x location of grid utilized to plot DepthEmbedding manifold.
            iy (int): x location of grid utilized to plot DepthEmbedding manifold.
            vmax (int): maximum value used to scale histogram image. Counts larger than \
                vmax saturate the histogram image. """
        # Visualize manifold: 
        if n_points is None or n_points is 0:
            n_points = self.n_training  
        Ix = ix*1.0 / self.nx * self.nx_resample
        Iy = iy*1.0 / self.ny * self.ny_resample
        calibration_data = self.get_data_grid(ix,iy) 
        filter = LikelihoodFilter(self.forward_model_2D)
        calibration_data_filtered, rejected = filter.filter(calibration_data, Ix,Iy, \
            points=n_points) 
        model_estimator = ModelDepthEmbedding(nz=self.nz, n_neighbours=self.n_neighbours)
        model_estimator.set_calibration_data(calibration_data_filtered)
        model_estimator.set_depth_prior(self.prior)
        model_estimator.estimate_forward_model() 
        model_estimator.visualize_manifold(nd=3)

        # Visualize histograms of the locations of interaction: 
        print "Visualizing histograms of reconstructed locations of interaction .. "
        if self.coordinates_centroid is not None: 
            self.histogram_centroid.show(vmax=vmax)
        if self.coordinates_2D_MAP_filtered is not None: 
            self.histogram_2D_MAP.show(vmax=vmax)
        if self.coordinates_3D_MAP_DE_filtered is not None: 
            self.histogram_3D_MAP_DE.show(axis=2, index=None, vmax=vmax)
        if self.coordinates_3D_MAP_MLEEM_filtered is not None: 
            self.histogram_3D_MAP_MLEEM.show(axis=2, index=None, vmax=vmax)
        
        # Visualize histograms of the energy: 
        print "Visualizing reconstructed energy spectra .. "
        fig = pl.figure()
        if self.spectrum_centroid is not None: 
            self.spectrum_centroid.show(fig=fig.number)
        if self.spectrum_2D_MAP is not None: 
            self.spectrum_2D_MAP.show(fig=fig.number)
        if self.spectrum_3D_MAP_DE is not None: 
            self.spectrum_3D_MAP_DE.show(fig=fig.number)
        if self.spectrum_3D_MAP_MLEEM is not None: 
            self.spectrum_3D_MAP_MLEEM.show(fig=fig.number)

        # Visualizing the depth distribution 
        print "Visualizing the depth distribution of the reconstructed events .."
        pl.figure()
        pl.plot(self.prior)
        pl.hold(1)
        
        h = self.histogram_3D_MAP_DE.histogram.sum(0).sum(0)
        h = (1.0*h) / h.sum()
        pl.plot(h)

        h = self.histogram_3D_MAP_MLEEM.histogram.sum(0).sum(0)
        h = (1.0*h) / h.sum()
        pl.plot(h)


    def run(self): 
        """Run all experiments: 
        - load experimental photon interaction data (perpendicular photon beam on grid)
        - compute forward models: 2D, 3D DepthEmbedding, 3D DepthEmbedding+MLEEM
        - resample forward models on higher resolution grid
        - define test grid (photon beams to test the performance of the recon algorithms)
        - reconstruct test data using Centroid, 3D max-a-posteriori, 3D max-a-posteriori 
            with DepthEmbedding forward model and with DepthEmbedding+MLEEM forward model
        - compute bias and variance of test data using each recon algorithm 
        - create images to display results (recon coordinates histograms, 
            energy histograms, DepthEmbedding manifold, distribition of recon depth of 
            interaction)."""
        print "-Estimating the forward model 2D .."
        if self.load_forward_model_2D() is None: 
            self.estimate_forward_model_2D() 
            self.save_forward_model_2D()
        print "-Estimating the forward model DepthEmbedding .."
        if self.load_forward_model_DE() is None: 
            self.estimate_forward_model_DE() 
            self.save_forward_model_DE() 
        print "-Estimating the forward model MLEEM .."
        if self.load_forward_model_MLEEM() is None:
            self.estimate_forward_model_MLEEM() 
            self.save_forward_model_MLEEM() 

        print "-Upsampling the forward models",\
            [self.nx_resample, self.ny_resample, self.nz_resample],".."
        self.resample_forward_model_2D([self.nx_resample, self.ny_resample]) 
        self.resample_forward_model_DE([self.nx_resample, self.ny_resample, \
            self.nz_resample]) 
        self.resample_forward_model_MLEEM([self.nx_resample, self.ny_resample, \
            self.nz_resample]) 
        prior_model = BeerLambert(self.scintillator_attenuation_coefficient, \
            self.nz_resample, self.scintillator_thickness/self.nz)
        self.prior = prior_model.get_probability_mass_function()

        print "-Loading 2D test grid"
        #self.load_test_grid([1,6,11,16,21,26,31], [1,6,11,16,21,26,31])
        self.load_test_grid([2,9,16,23,30], [2,9,16,23,30])
        print "-Reconstruction using centroid algorithm"
        self.reconstruct_grid_centroid(shiftx=2.8, shifty=2.8, \
            scale=5.2*self.nx_resample / 32.0)
        self.make_histograms_grid_centroid()
        print "-Reconstruction using 2D maximum-a-posteriori"
        if self.load_reconstructions_grid_2D_MAP() is None: 
            self.reconstruct_grid_2D_MAP()
            self.save_reconstructions_grid_2D_MAP()
        self.filter_events_2D_MAP(fraction=0.7) 
        self.make_histograms_grid_2D_MAP()
        print "-Reconstruction using 3D maximum-a-posteriori (DepthEmbedding model)"
        if self.load_reconstructions_grid_3D_MAP_DE() is None: 
            self.reconstruct_grid_3D_MAP_DE()
            self.save_reconstructions_grid_3D_MAP_DE()
        self.filter_events_3D_MAP_DE(fraction=0.7) 
        self.make_histograms_grid_3D_MAP_DE()
        print "-Reconstruction using 3D maximum-a-posteriori (DepthEmbedding+MLEE model)"
        if self.load_reconstructions_grid_3D_MAP_MLEEM() is None: 
            self.reconstruct_grid_3D_MAP_MLEEM()
            self.save_reconstructions_grid_3D_MAP_MLEEM()
        self.filter_events_3D_MAP_MLEEM(fraction=0.7) 
        self.make_histograms_grid_3D_MAP_MLEEM()
        print "Computing Bias and Variance"
        self.compute_bias_and_variance_grid() 
        print "Visualizing results"
        self.visualize_results_grid() 
        print "TestCmice Done"

    def get_data_grid(self, x,y):
        """Load cMiCe perpendicular photon beam experimental data from files.  
        
        Args: 
            x (int): x index of beam position. 
            y (int): y index of beam position. """
        return get_data_cmice_grid(x,y, path=self.input_data_path)

    def _print_percentage(self,value,max_value):
        """Print progress percentage to stdout."""
        print "Progress: %d%%"%int32(100*value/(max_value-1))






if __name__ == "__main__": 
    print "--- DepthEmbedding ---"
    print "-- Calibration of cMiCE PET camera with 8mm-thick crystal, "+ \
        "without resampling of the forward model.."
    cmice = Cmice_8mm(nx_resample=32, ny_resample=32, nz_resample=16, n_neighbours=12, \
        n_training=2500, nz=16)
    cmice.run() 
    
    print "-- Calibration of cMiCE PET camera with 8mm-thick crystal, "+ \
        "with resampling of the forward model.."
    cmice = Cmice_8mm(nx_resample=128, ny_resample=128, nz_resample=32, n_neighbours=12, \
        n_training=2500, nz=32)
    cmice.run() 
    print "-- Done"

