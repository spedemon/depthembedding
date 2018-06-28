# -*- coding: utf-8 -*-
# Harvard Medical School, Martinos Center for Biomedical Imaging 
# Aalto University, Department of Computer Science 

"""Calibrate cMiCE (Continuous Miniature Crystal Element, University of Washington) 
gamma detector with 15mm scintillation crystal, using the DepthEmbedding algorithm [1] 
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
from numpy import exp, convolve, sqrt, mgridimport scipy.ndimage.filters
import numpy as np
import copy
import os 
import pylab as pl


BATCH_SIZE = 256  # Number of events reconstructed in parallel (i.e. in a single batch). 
                  # If set to 0, all events are reconstructed at once: faster but may 
                  # starve memory when using large reconstruction grids 
                  # (e.g. when up-sampling the forward model). 


def get_data_cmice_grid(x,y,path="./20090416_AA1003_cmice_data_15mm/"): 
    """Load cMiCe 15mm-crystal perpendicular photon beam experimental data. 
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
            Defaults to './20090416_AA1003_cmice_data_15mm/'. 
            
    Returns: 
        ndarray: 2D photon interaction float matrix of sixe (N_events, N_detectors), \
            where N_events is the number of interaction events recorded 
            at the x,y grid location and N_detectors is the number of photomultiplier 
            tubes of cMiCe (i.e. 36). """
    filename = path + "%0.3f"%(1013*x/1000.0)+"_"+"%0.3f"%(1013*y/1000.0)+".mat" 
    import h5py
    print filename
    data = np.array(h5py.File(filename)['data']).T
    #data = scipy.io.loadmat(filename)['data']
    data[data<0] = 0
    return float32(data.T)



def get_data_cmice_45_degrees(index, path="./"):
    """cMiCe: data acquired with 45 degrees beam."""
    path = path + "20090416_AA1003_cmice_data_15mm_45_deg/"
    filenames = os.listdir(path)
    if index>=len(filenames): 
        return None
    filename = path+filenames[index]
    data = scipy.io.loadmat(filename)['data']
    data[data<0] = 0
    return float32(data.T)



class Cmice_15mm(): 
    """Estimation of the forward model of the cMiCe PET camera and reconstruction using various 2D and 3D methods. """
    def __init__(self, n_neighbours=12, n_training=2500, nz=20, nx_resample=26, ny_resample=49, nz_resample=20): 
        self.nx = 26
        self.ny = 49 
        self.nx_resample = nx_resample
        self.ny_resample = ny_resample
        self.nz_resample = nz_resample
        self.n_detectors = 64 
        self.scintillator_attenuation_coefficient = 0.83  #[cm-1]
        self.scintillator_thickness = 1.5                 #[cm]
        self.input_data_path = "./input_data_cmice_15mm/"
        self.results_path    = "./results_cmice_15mm/"
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
        prior_model = BeerLambert(alpha_inv_cm=self.scintillator_attenuation_coefficient, n_bins=self.nz, bin_size_cm=self.scintillator_thickness/self.nz)
        self.prior = prior_model.get_probability_mass_function()

    def load_forward_model_2D(self, filename='/model_cmice_2D.mat'): 
        filename = self.results_path+filename
        try: 
            model = scipy.io.loadmat(filename)['model_cmice_2D']
        except: 
            return None
        print "-Found 2D forward model file: %s - not recomputing it."%filename
        self.forward_model_2D = model
        return model

    def load_forward_model_DE(self, filename='/model_cmice_DE.mat'): 
        filename = self.results_path+filename
        try: 
            model = scipy.io.loadmat(filename)['model_cmice_DE']
        except: 
            return None
        print "-Found 3D DepthEmbedding forward model file: %s - not recomputing it."%filename
        self.forward_model_DE = model
        return model

    def load_forward_model_MLEEM(self, filename='/model_cmice_MLEEM.mat'): 
        filename = self.results_path+filename
        try: 
            model = scipy.io.loadmat(filename)['model_cmice_MLEEM']
        except: 
            return None
        print "-Found 3D DepthEmbedding+MLEEM forward model file: %s - not recomputing it."%filename
        self.forward_model_MLEEM = model
        return model


    def save_forward_model_2D(self, filename='/model_cmice_2D.mat'):
        filename = self.results_path+filename 
        scipy.io.savemat(filename, {'model_cmice_2D':self.forward_model_2D})

    def save_forward_model_DE(self, filename='/model_cmice_DE.mat'):
        filename = self.results_path+filename 
        scipy.io.savemat(filename, {'model_cmice_DE':self.forward_model_DE})
 
    def save_forward_model_MLEEM(self, filename='/model_cmice_MLEEM.mat'): 
        filename = self.results_path+filename
        scipy.io.savemat(filename, {'model_cmice_MLEEM':self.forward_model_MLEEM})


    def load_reconstructions_grid_2D_MAP(self, filename='/coordinates_2D_MAP'): 
        filename = self.results_path+filename
        filename = filename + '_%d_%d_%d.mat'%(self.nx_resample, self.ny_resample, self.nz_resample)
        try: 
            coords = scipy.io.loadmat(filename)['coordinates_2D_MAP']
            energy = scipy.io.loadmat(filename)['energy_2D_MAP']
            likelihood = scipy.io.loadmat(filename)['likelihood_2D_MAP']
        except: 
            return None
        print "-Found 2D MAP reconstructed coordinates file: %s - not recomputing them."%filename
        for ix in range(len(coords)): 
            for iy in range(len(coords[0])):
                # for some reason, scipy.io.savemat adds an extra layer in the list, get read of it: 
                coords[ix][iy][0] = coords[ix][iy][0][0]
                coords[ix][iy][1] = coords[ix][iy][1][0]
                energy[ix][iy] = energy[ix][iy][0]
                likelihood[ix][iy] = likelihood[ix][iy][0]
        self.coordinates_2D_MAP = coords
        self.energy_2D_MAP = energy
        self.likelihood_2D_MAP = likelihood
        return coords, energy, likelihood

    def load_reconstructions_grid_3D_MAP_DE(self, filename='/coordinates_3D_MAP_DE'): 
        filename = self.results_path+filename
        filename = filename + '_%d_%d_%d.mat'%(self.nx_resample, self.ny_resample, self.nz_resample)
        try: 
            coords = scipy.io.loadmat(filename)['coordinates_3D_MAP_DE']
            energy = scipy.io.loadmat(filename)['energy_3D_MAP_DE']
            likelihood = scipy.io.loadmat(filename)['likelihood_3D_MAP_DE']
        except: 
            return None
        print "-Found 3D MAP DepthEmbedding reconstructed coordinates file: %s - not recomputing them."%filename
        for ix in range(len(coords)): 
            for iy in range(len(coords[0])):
                # for some reason, scipy.io.savemat adds an extra layer in the list, get read of it: 
                coords[ix][iy][0] = coords[ix][iy][0][0]
                coords[ix][iy][1] = coords[ix][iy][1][0]
                coords[ix][iy][2] = coords[ix][iy][2][0]
                energy[ix][iy] = energy[ix][iy][0]
                likelihood[ix][iy] = likelihood[ix][iy][0]
        self.coordinates_3D_MAP_DE = coords
        self.energy_3D_MAP_DE = energy
        self.likelihood_3D_MAP_DE = likelihood
        return coords, energy, likelihood

    def load_reconstructions_grid_3D_MAP_MLEEM(self, filename='/coordinates_3D_MAP_MLEEM'): 
        filename = self.results_path+filename
        filename = filename + '_%d_%d_%d.mat'%(self.nx_resample, self.ny_resample, self.nz_resample)
        try: 
            coords = scipy.io.loadmat(filename)['coordinates_3D_MAP_MLEEM']
            energy = scipy.io.loadmat(filename)['energy_3D_MAP_MLEEM']
            likelihood = scipy.io.loadmat(filename)['likelihood_3D_MAP_MLEEM']
        except: 
            return None
        print "-Found 3D MAP DepthEmbedding+MLEEM reconstructed coordinates file: %s - not recomputing them."%filename
        for ix in range(len(coords)): 
            for iy in range(len(coords[0])):
                # for some reason, scipy.io.savemat adds an extra layer in the list, get read of it: 
                coords[ix][iy][0] = coords[ix][iy][0][0]
                coords[ix][iy][1] = coords[ix][iy][1][0]
                coords[ix][iy][2] = coords[ix][iy][2][0]
                energy[ix][iy] = energy[ix][iy][0]
                likelihood[ix][iy] = likelihood[ix][iy][0]
        self.coordinates_3D_MAP_MLEEM = coords
        self.energy_3D_MAP_MLEEM = energy
        self.likelihood_3D_MAP_MLEEM = likelihood
        return coords, energy, likelihood


    def load_reconstructions_45_degrees_2D_MAP(self, filename='/coordinates_45_degrees_2D_MAP'): 
        filename = self.results_path+filename
        filename = filename + '_%d_%d_%d.mat'%(self.nx_resample, self.ny_resample, self.nz_resample)
        try: 
            coords = scipy.io.loadmat(filename)['coordinates_45_degrees_2D_MAP']
            energy = scipy.io.loadmat(filename)['energy_45_degrees_2D_MAP']
            #likelihood = scipy.io.loadmat(filename)['likelihood_45_degrees_2D_MAP']
        except: 
            return None
        print "-Found 45_degrees 2D MAP reconstructed coordinates file: %s - not recomputing them."%filename
        self.coordinates_45_degrees_2D_MAP = coords
        self.energy_45_degrees_2D_MAP = energy
        #self.likelihood_45_degrees_2D_MAP = likelihood
        return coords, energy

    def load_reconstructions_45_degrees_3D_MAP_DE(self, filename='/coordinates_45_degrees_3D_MAP_DE'): 
        filename = self.results_path+filename
        filename = filename + '_%d_%d_%d.mat'%(self.nx_resample, self.ny_resample, self.nz_resample)
        try: 
            coords = scipy.io.loadmat(filename)['coordinates_45_degrees_3D_MAP_DE']
            energy = scipy.io.loadmat(filename)['energy_45_degrees_3D_MAP_DE']
            #likelihood = scipy.io.loadmat(filename)['likelihood_45_degrees_3D_MAP_DE']
        except: 
            return None
        print "-Found 45_degrees 3D MAP DepthEmbedding reconstructed coordinates file: %s - not recomputing them."%filename
        self.coordinates_45_degrees_3D_MAP_DE = coords
        self.energy_45_degrees_3D_MAP_DE = energy
        #self.likelihood_45_degrees_3D_MAP_DE = likelihood
        return coords, energy

    def load_reconstructions_45_degrees_3D_MAP_MLEEM(self, filename='/coordinates_45_degrees_3D_MAP_MLEEM'): 
        filename = self.results_path+filename
        filename = filename + '_%d_%d_%d.mat'%(self.nx_resample, self.ny_resample, self.nz_resample)
        try: 
            coords = scipy.io.loadmat(filename)['coordinates_45_degrees_3D_MAP_MLEEM']
            energy = scipy.io.loadmat(filename)['energy_45_degrees_3D_MAP_MLEEM']
            #likelihood = scipy.io.loadmat(filename)['likelihood_45_degrees_3D_MAP_MLEEM']
        except: 
            return None
        print "-Found 45_degrees 3D MAP DepthEmbedding+MLEEM reconstructed coordinates file: %s - not recomputing them."%filename
        self.coordinates_45_degrees_3D_MAP_MLEEM = coords
        self.energy_45_degrees_3D_MAP_MLEEM = energy
        #self.likelihood_45_degrees_3D_MAP_MLEEM = likelihood
        return coords, energy


    def save_reconstructions_grid_2D_MAP(self, filename='/coordinates_2D_MAP'): 
        filename = self.results_path+filename
        filename = filename + '_%d_%d_%d.mat'%(self.nx_resample, self.ny_resample, self.nz_resample)
        scipy.io.savemat(filename, {'coordinates_2D_MAP':self.coordinates_2D_MAP,'energy_2D_MAP':self.energy_2D_MAP,'likelihood_2D_MAP':self.likelihood_2D_MAP})

    def save_reconstructions_grid_3D_MAP_DE(self, filename='/coordinates_3D_MAP_DE'): 
        filename = self.results_path+filename
        filename = filename + '_%d_%d_%d.mat'%(self.nx_resample, self.ny_resample, self.nz_resample)
        scipy.io.savemat(filename, {'coordinates_3D_MAP_DE':self.coordinates_3D_MAP_DE,'energy_3D_MAP_DE':self.energy_3D_MAP_DE,'likelihood_3D_MAP_DE':self.likelihood_3D_MAP_DE})

    def save_reconstructions_grid_3D_MAP_MLEEM(self, filename='/coordinates_3D_MAP_MLEEM'): 
        filename = self.results_path+filename
        filename = filename + '_%d_%d_%d.mat'%(self.nx_resample, self.ny_resample, self.nz_resample)
        scipy.io.savemat(filename, {'coordinates_3D_MAP_MLEEM':self.coordinates_3D_MAP_MLEEM,'energy_3D_MAP_MLEEM':self.energy_3D_MAP_MLEEM,'likelihood_3D_MAP_MLEEM':self.likelihood_3D_MAP_MLEEM})


    def save_reconstructions_45_degrees_2D_MAP(self, filename='/coordinates_45_degrees_2D_MAP'): 
        filename = self.results_path+filename
        filename = filename + '_%d_%d_%d.mat'%(self.nx_resample, self.ny_resample, self.nz_resample)
        #scipy.io.savemat(filename, {'coordinates_45_degrees_2D_MAP':self.coordinates_45_degrees_2D_MAP,'energy_45_degrees_2D_MAP':self.energy_45_degrees_2D_MAP,'likelihood_45_degrees_2D_MAP':self.likelihood_45_degrees_2D_MAP)
        scipy.io.savemat(filename, {'coordinates_45_degrees_2D_MAP':self.coordinates_45_degrees_2D_MAP,'energy_45_degrees_2D_MAP':self.energy_45_degrees_2D_MAP})

    def save_reconstructions_45_degrees_3D_MAP_DE(self, filename='/coordinates_45_degrees_3D_MAP_DE'):
        filename = self.results_path+filename 
        filename = filename + '_%d_%d_%d.mat'%(self.nx_resample, self.ny_resample, self.nz_resample)
        #scipy.io.savemat(filename, {'coordinates_45_degrees_3D_MAP_DE':self.coordinates_45_degrees_3D_MAP_DE,'energy_45_degrees_3D_MAP_DE':self.energy_45_degrees_3D_MAP_DE,'likelihood_45_degrees_3D_MAP_DE':self.likelihood_45_degrees_3D_MAP_DE})
        scipy.io.savemat(filename, {'coordinates_45_degrees_3D_MAP_DE':self.coordinates_45_degrees_3D_MAP_DE,'energy_45_degrees_3D_MAP_DE':self.energy_45_degrees_3D_MAP_DE})        

    def save_reconstructions_45_degrees_3D_MAP_MLEEM(self, filename='/coordinates_45_degrees_3D_MAP_MLEEM'): 
        filename = self.results_path+filename
        filename = filename + '_%d_%d_%d.mat'%(self.nx_resample, self.ny_resample, self.nz_resample)
        #scipy.io.savemat(filename, {'coordinates_45_degrees_3D_MAP_MLEEM':self.coordinates_45_degrees_3D_MAP_MLEEM,'energy_45_degrees_3D_MAP_MLEEM':self.energy_45_degrees_3D_MAP_MLEEM,'likelihood_45_degrees_3D_MAP_MLEEM':self.likelihood_45_degrees_3D_MAP_MLEEM})
        scipy.io.savemat(filename, {'coordinates_45_degrees_3D_MAP_MLEEM':self.coordinates_45_degrees_3D_MAP_MLEEM,'energy_45_degrees_3D_MAP_MLEEM':self.energy_45_degrees_3D_MAP_MLEEM})
        

    def estimate_forward_model_2D(self): 
        model = zeros([self.nx, self.ny, self.n_detectors])
        for ix in range(self.nx): 
            #self.print_percentage(ix,self.nx)
            for iy in range(self.ny): 
                self.print_percentage(iy+ix*self.ny,self.ny*self.nx)
                calibration_data = self.get_data_grid(ix,iy)
                model_estimator = Model2D()
                model_estimator.set_calibration_data(calibration_data)
                model[ix,iy,:] = model_estimator.estimate_forward_model()
        self.forward_model_2D = model 
        return self.forward_model_2D

    def estimate_forward_model_DE(self): 
        model = zeros([self.nx, self.ny, self.nz, self.n_detectors])
        for ix in range(self.nx): 
            #self.print_percentage(ix,self.nx)
            for iy in range(self.ny): 
                self.print_percentage(iy+ix*self.ny,self.ny*self.nx)
                calibration_data = self.get_data_grid(ix,iy)
                filter = LikelihoodFilter(self.forward_model_2D)
                calibration_data_filtered, rejected = filter.filter(calibration_data, ix,iy, method="near_coordinates", points=self.n_training) 
                print "Number of points for location %d,%d:"%(ix,iy),calibration_data_filtered.shape[0]
                model_estimator = ModelDepthEmbedding(nz=self.nz, n_neighbours=self.n_neighbours)
                model_estimator.set_calibration_data(calibration_data_filtered)
                model_estimator.set_depth_prior(self.prior) 
                m = model_estimator.estimate_forward_model(unit_norm=False, zero_mean=False) 
                print "Max expected signal:    z=0: %2.2f    z=%d: %2.2f"%(m[0,:].max(), self.nz-1, m[self.nz-1,:].max())
                model[ix,iy,:,:] = m 
        self.forward_model_DE = model 
        return self.forward_model_DE
    
    def estimate_forward_model_MLEEM(self, sigma_smoothness=0.0, iterations_smoothness=5): 
        model = zeros([self.nx, self.ny, self.nz, self.n_detectors])
        for ix in range(self.nx): 
            for iy in range(self.ny): 
                self.print_percentage(iy+ix*self.ny,self.ny*self.nx)
                calibration_data = self.get_data_grid(ix,iy)
                filter = LikelihoodFilter(self.forward_model_2D)
                calibration_data_filtered, rejected = filter.filter(calibration_data, ix,iy, points=self.n_training) 
                model_estimator = ModelMLEEM(initial_model=self.forward_model_DE[ix,iy,:,:])
                model_estimator.set_calibration_data(calibration_data_filtered) 
                model_estimator.set_depth_prior(self.prior) 
                model[ix,iy,:,:] = model_estimator.estimate_forward_model(n_max_iterations=20, method='soft', smoothness=0.0, prune=False) 
        self.forward_model_MLEEM = model
        if sigma_smoothness > 1e-9: 
            print "MLE-EM smoothness refinement .."
            for ii in range(iterations_smoothness): 
                model_smoothed = scipy.ndimage.filters.gaussian_filter(model, sigma=sigma_smoothness, mode='reflect') 
                for ix in range(self.nx): 
                    for iy in range(self.ny): 
                        self.print_percentage(iy+ix*self.ny+ii*self.ny*self.nx,iterations_smoothness*self.ny*self.nx)
                        calibration_data = self.get_data_grid(ix,iy)
                        filter = LikelihoodFilter(self.forward_model_2D)
                        calibration_data_filtered, rejected = filter.filter(calibration_data, ix,iy, points=self.n_training) 
                        model_estimator = ModelMLEEM(initial_model=model_smoothed[ix,iy,:,:])
                        model_estimator.set_calibration_data(calibration_data_filtered) 
                        model_estimator.set_depth_prior(self.prior) 
                        model[ix,iy,:,:] = model_estimator.estimate_forward_model() 
            self.forward_model_MLEEM = model   
        return self.forward_model_MLEEM


    def resample_forward_model_2D(self, grid_size, interpolation='cubic', noise_sigma=0.3): 
        if self.nx == self.nx_resample and self.ny == self.ny_resample and self.nz == self.nz_resample: 
            print "Not resampling 2D model."
            return self.forward_model_2D
        interpolator = ModelInterpolator()
        forward_model = interpolator.resample(self.forward_model_2D, grid_size, interpolation=interpolation)
        if noise_sigma > 1e-9:
            # A small amount of noise in the forward model effectively removes interpolation artefacts 
            forward_model = np.random.normal(forward_model, noise_sigma)
            forward_model[forward_model<=0.0]=0.0
        self.forward_model_2D = forward_model
        return forward_model

    def resample_forward_model_DE(self, grid_size, interpolation='cubic', noise_sigma=0.3): 
        if self.nx == self.nx_resample and self.ny == self.ny_resample and self.nz == self.nz_resample: 
            print "Not resampling DE model."
            return self.forward_model_DE
        interpolator = ModelInterpolator()
        forward_model = interpolator.resample(self.forward_model_DE, grid_size, interpolation=interpolation)
        if noise_sigma > 1e-9:
            # A small amount of noise in the forward model effectively removes interpolation artefacts 
            forward_model = np.random.normal(forward_model, noise_sigma)
            forward_model[forward_model<=0.0]=0.0
        self.forward_model_DE = forward_model
        return forward_model

    def resample_forward_model_MLEEM(self, grid_size, interpolation='cubic', noise_sigma=0.3): 
        if self.nx == self.nx_resample and self.ny == self.ny_resample and self.nz == self.nz_resample: 
            print "Not resampling MLEEM model."
            return self.forward_model_MLEEM
        interpolator = ModelInterpolator()
        forward_model = interpolator.resample(self.forward_model_MLEEM, grid_size, interpolation=interpolation)
        if noise_sigma > 1e-9:
            # A small amount of noise in the forward model effectively removes interpolation artefacts 
            forward_model = np.random.normal(forward_model, noise_sigma)
            forward_model[forward_model<=0.0]=0.0
        self.forward_model_MLEEM = forward_model
        return forward_model

    def load_test_grid(self, x_locations, y_locations): 
        self.grid = []
        for ix in x_locations: 
            grid_row = [] 
            for iy in y_locations: 
                data = self.get_data_grid(ix,iy)
                grid_row.append(data)
            self.grid.append(grid_row) 
        self.grid_shape = (len(x_locations),len(y_locations))
        print self.grid_shape
        self.grid_locations = [tile(x_locations,(len(y_locations),1)).transpose(), np.tile(np.int32(y_locations).reshape((len(y_locations),1)),(1,len(x_locations))).transpose()]
        return self.grid

    def load_test_45_degrees(self, indexes=None): 
        self.data_45_degrees = []
        if indexes is None: 
            index = -1 
            while 1:
                index += 1 
                data = self.get_data_45_degrees(index)
                if data is None: 
                    break 
                self.data_45_degrees.append(data)
        else: 
            for index in indexes: 
                data = self.get_data_45_degrees(index)
                self.data_45_degrees.append(data) 


            
    def reconstruct_grid_centroid(self, shiftx=2.9, shifty=2.5, scale=8.0): 
        self.coordinates_centroid = [] 
        self.energy_centroid = []
        x_detectors = tile(linspace(0,9,8),(1,8))[0] - 4.5 
        y_detectors = repeat(linspace(0,9,8),8,axis=0) - 4.3
        reconstructor = ReconstructorCentroid(x_detectors=x_detectors, y_detectors=y_detectors, x_max=self.ny_resample-1, y_max=self.nx_resample-1, shiftx=shiftx, shifty=shifty, scale=scale, exponent=1.0)  
        for ix in range(self.grid_shape[0]): 
            row_coordinates = [] 
            row_energy = []
            for iy in range(self.grid_shape[1]):
                data = self.grid[ix][iy]
                [yrec,xrec,energyrec] = reconstructor.reconstruct(data) 
                xrec = self.nx_resample - 1.0 - xrec
                yrec = self.ny_resample - 1.0 - yrec
                row_coordinates.append([xrec,yrec])
                row_energy.append(energyrec)
            self.coordinates_centroid.append(row_coordinates)
            self.energy_centroid.append(row_energy) 
        return self.coordinates_centroid

    def make_histograms_grid_centroid(self): 
        self.histogram_centroid = HistogramCoordinates(self.nx_resample, self.ny_resample)
        self.spectrum_centroid = EnergySpectrum(scale="auto", peak=511.0, max_energy=1500) 
        for ix in range(self.grid_shape[0]): 
            for iy in range(self.grid_shape[1]):
                [xrec,yrec] = self.coordinates_centroid[ix][iy]
                energyrec = self.energy_centroid[ix][iy]
                self.histogram_centroid.add_data([xrec,yrec])
                self.spectrum_centroid.add_data(energyrec)

    def reconstruct_grid_2D_MAP(self): 
        self.coordinates_2D_MAP = []
        self.energy_2D_MAP = []
        self.likelihood_2D_MAP = []
        reconstructor = ReconstructorMAP(self.forward_model_2D) 
         
        reconstructor.set_unit_norm(1)
        #reconstructor.set_zero_mean(1)
        #reconstructor.set_unit_norm_fw(1)
        #reconstructor.set_zero_mean_fw(1)
        
        for ix in range(self.grid_shape[0]): 
            #self.print_percentage(ix,self.grid_shape[0])
            row_coordinates = []
            row_energy = []
            row_likelihood = []
            for iy in range(self.grid_shape[1]):
                self.print_percentage(iy+ix*self.grid_shape[1],self.grid_shape[0]*self.grid_shape[1])
                data = self.grid[ix][iy]
                [xrec,yrec,energyrec,logposterior] = reconstructor.reconstruct(data, batch_size=BATCH_SIZE) 
                row_coordinates.append([xrec,yrec])
                row_energy.append(energyrec)
                row_likelihood.append(logposterior)
            self.coordinates_2D_MAP.append(row_coordinates)
            self.energy_2D_MAP.append(row_energy)
            self.likelihood_2D_MAP.append(row_likelihood)
        return self.coordinates_2D_MAP
    
    def make_histograms_grid_2D_MAP(self): 
        self.histogram_2D_MAP = HistogramCoordinates(self.nx_resample, self.ny_resample)
        self.spectrum_2D_MAP = EnergySpectrum(scale="auto", peak=511.0, max_energy=1500) 
        for ix in range(self.grid_shape[0]): 
            for iy in range(self.grid_shape[1]):
                [xrec,yrec] = self.coordinates_2D_MAP_filtered[ix][iy]
                energyrec = self.energy_2D_MAP_filtered[ix][iy]
                self.histogram_2D_MAP.add_data([xrec,yrec])
                self.spectrum_2D_MAP.add_data(energyrec)

    def reconstruct_grid_3D_MAP_DE(self): 
        self.coordinates_3D_MAP_DE = []
        self.energy_3D_MAP_DE = []
        self.likelihood_3D_MAP_DE = []
        reconstructor = ReconstructorMAP(self.forward_model_DE)  
        reconstructor.set_prior(self.prior)
        for ix in range(self.grid_shape[0]): 
            #self.print_percentage(ix,self.grid_shape[0])
            row_coordinates = []
            row_energy = []
            row_likelihood = []
            for iy in range(self.grid_shape[1]):
                self.print_percentage(iy+ix*self.grid_shape[1],self.grid_shape[0]*self.grid_shape[1])
                data = self.grid[ix][iy]
                [xrec,yrec,zrec,energyrec,logposterior] = reconstructor.reconstruct(data, batch_size=BATCH_SIZE) 
                row_coordinates.append([xrec,yrec,zrec])
                row_energy.append(energyrec)
                row_likelihood.append(logposterior)
            self.coordinates_3D_MAP_DE.append(row_coordinates)
            self.energy_3D_MAP_DE.append(row_energy)
            self.likelihood_3D_MAP_DE.append(row_likelihood)
        return self.coordinates_3D_MAP_DE

    def make_histograms_grid_3D_MAP_DE(self):
        self.histogram_3D_MAP_DE = HistogramCoordinates(self.nx_resample, self.ny_resample, self.nz_resample)
        self.spectrum_3D_MAP_DE = EnergySpectrum(scale="auto", peak=511.0, max_energy=1500) 
        for ix in range(self.grid_shape[0]): 
            for iy in range(self.grid_shape[1]):
                [xrec,yrec,zrec] = self.coordinates_3D_MAP_DE_filtered[ix][iy]
                energyrec = self.energy_3D_MAP_DE_filtered[ix][iy]
                self.histogram_3D_MAP_DE.add_data([xrec,yrec,zrec])
                self.spectrum_3D_MAP_DE.add_data(energyrec)

    def reconstruct_grid_3D_MAP_MLEEM(self): 
        self.coordinates_3D_MAP_MLEEM = []
        self.energy_3D_MAP_MLEEM = []
        self.likelihood_3D_MAP_MLEEM = []
        reconstructor = ReconstructorMAP(self.forward_model_MLEEM)  
        reconstructor.set_prior(self.prior)
        for ix in range(self.grid_shape[0]): 
            #self.print_percentage(ix,self.grid_shape[0])
            row_coordinates = []
            row_energy = []
            row_likelihood = []
            for iy in range(self.grid_shape[1]):
                self.print_percentage(iy+ix*self.grid_shape[1],self.grid_shape[0]*self.grid_shape[1])
                data = self.grid[ix][iy]
                [xrec,yrec,zrec,energyrec,logposterior] = reconstructor.reconstruct(data, batch_size=BATCH_SIZE) 
                row_coordinates.append([xrec,yrec,zrec])
                row_energy.append(energyrec)
                row_likelihood.append(logposterior)
            self.coordinates_3D_MAP_MLEEM.append(row_coordinates)
            self.energy_3D_MAP_MLEEM.append(row_energy)
            self.likelihood_3D_MAP_MLEEM.append(row_likelihood)
        return self.coordinates_3D_MAP_MLEEM

    def make_histograms_grid_3D_MAP_MLEEM(self):
        self.histogram_3D_MAP_MLEEM = HistogramCoordinates(self.nx_resample, self.ny_resample, self.nz_resample)
        self.spectrum_3D_MAP_MLEEM = EnergySpectrum(scale="auto", peak=511.0, max_energy=1500) 
        for ix in range(self.grid_shape[0]): 
            for iy in range(self.grid_shape[1]):
                [xrec,yrec,zrec] = self.coordinates_3D_MAP_MLEEM_filtered[ix][iy]
                energyrec = self.energy_3D_MAP_MLEEM_filtered[ix][iy]
                self.histogram_3D_MAP_MLEEM.add_data([xrec,yrec,zrec])
                self.spectrum_3D_MAP_MLEEM.add_data(energyrec)

    def filter_events_2D_MAP(self, fraction=0.7): 
        self.coordinates_2D_MAP_filtered = []
        self.energy_2D_MAP_filtered = [] 
        self.likelihood_2D_MAP_filtered = []
        for ix in range(self.grid_shape[0]): 
            C = []
            E = []
            L = []
            for iy in range(self.grid_shape[1]):
                N = np.int32(len(self.coordinates_2D_MAP[ix][iy][0])*fraction)
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

    def filter_events_3D_MAP_DE(self, fraction=0.7): 
        self.coordinates_3D_MAP_DE_filtered = []
        self.energy_3D_MAP_DE_filtered = []
        self.likelihood_3D_MAP_DE_filtered = []
        for ix in range(self.grid_shape[0]): 
            C = []
            E = []
            L = []
            for iy in range(self.grid_shape[1]):
                N = np.int32(len(self.coordinates_3D_MAP_DE[ix][iy][0])*fraction)
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
        
    def filter_events_3D_MAP_MLEEM(self, fraction=0.7): 
        self.coordinates_3D_MAP_MLEEM_filtered = []
        self.energy_3D_MAP_MLEEM_filtered = []
        self.likelihood_3D_MAP_MLEEM_filtered = []
        for ix in range(self.grid_shape[0]): 
            C = []
            E = []
            L = []
            for iy in range(self.grid_shape[1]):
                N = np.int32(len(self.coordinates_3D_MAP_MLEEM[ix][iy][0])*fraction)
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

    def scale_grid_locations(self):
        n_points_x = self.grid_locations[0].shape[0]
        n_points_y = self.grid_locations[0].shape[1]
        grid_locations_scaled = copy.deepcopy(self.grid_locations)
        for ix in range(n_points_x): 
            for iy in range(n_points_y): 
                grid_locations_scaled[0][ix][iy] = grid_locations_scaled[0][ix][iy] *1.0 / self.nx * self.nx_resample
                grid_locations_scaled[1][ix][iy] = grid_locations_scaled[1][ix][iy] *1.0 / self.ny * self.ny_resample
        return grid_locations_scaled

    def compute_bias_and_variance_grid(self): 
        grid_locations = self.scale_grid_locations() 
        if self.coordinates_centroid is not None: 
            stats_centroid = StatsCoordinates(self.coordinates_centroid, grid_locations, [self.nx_resample,self.ny_resample], "Centroid", scale=(25.0/self.nx_resample, 50.0/self.ny_resample, 15.0/self.nz_resample))
            stats_centroid.print_summary()
        if self.coordinates_2D_MAP_filtered is not None: 
            stats_2D_MAP = StatsCoordinates(self.coordinates_2D_MAP_filtered, grid_locations, [self.nx_resample,self.ny_resample], "2D MAP", scale=(25.0/self.nx_resample, 50.0/self.ny_resample, 15.0/self.nz_resample))
            stats_2D_MAP.print_summary()
        if self.coordinates_3D_MAP_DE_filtered is not None: 
            stats_3D_MAP_DE = StatsCoordinates(self.coordinates_3D_MAP_DE_filtered, grid_locations, [(0,self.nx_resample),(0,self.ny_resample),(2,self.nz_resample-2)], "3D MAP DepthEmbedding", scale=(25.0/self.nx_resample, 50.0/self.ny_resample, 15.0/self.nz_resample))
            stats_3D_MAP_DE.print_summary()
            #stats_3D_MAP_DE.plot_depth_density(6)
        if self.coordinates_3D_MAP_MLEEM_filtered is not None: 
            stats_3D_MAP_MLEEM = StatsCoordinates(self.coordinates_3D_MAP_MLEEM_filtered, grid_locations, [(0,self.nx_resample),(0,self.ny_resample),(2,self.nz_resample-2)], "3D MAP DepthEmbedding+MLEEM", scale=(25.0/self.nx_resample, 50.0/self.ny_resample, 15.0/self.nz_resample))
            stats_3D_MAP_MLEEM.print_summary()
            #stats_3D_MAP_MLEEM.plot_depth_density(6)
            

    def visualize_results_grid(self, n_points=5000, vmax=3000): 
        # Visualize manifold: 
        print "Visualizing manifold .."
        if n_points is None or n_points is 0.0: 
            n_points = self.n_training 
        ix = 15
        iy = 15
        Ix = ix*1.0 / self.nx * self.nx_resample
        Iy = iy*1.0 / self.ny * self.ny_resample
        calibration_data = self.get_data_grid(ix,iy) 
        filter = LikelihoodFilter(self.forward_model_2D)
        calibration_data_filtered, rejected = filter.filter(calibration_data, Ix, Iy, method="near_coordinates", points=self.n_training) 

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


    def reconstruct_45_degrees_centroid(self, shiftx=3.6,shifty=3.5, scale=6.5): 
        self.coordinates_45_degrees_centroid = [] 
        self.energy_45_degrees_centroid = []
        x_detectors = tile(linspace(0,9,8),(1,8))[0] - 4.5 
        y_detectors = repeat(linspace(0,9,8),8,axis=0) - 4.3 
        reconstructor = ReconstructorCentroid(x_detectors=x_detectors, y_detectors=y_detectors, x_max=self.nx-1, y_max=self.ny-1, shiftx=shiftx, shifty=shifty, scale=scale, exponent=1.0)  
        for index in range(len(self.data_45_degrees)): 
            data = self.data_45_degrees[index] 
            [xrec,yrec,energyrec] = reconstructor.reconstruct(data) 

            self.coordinates_45_degrees_centroid.append([xrec,yrec]) 
            self.energy_45_degrees_centroid.append(energyrec) 
        return self.coordinates_45_degrees_centroid 

    def make_histograms_45_degrees_centroid(self): 
        self.histogram_45_degrees_centroid = HistogramCoordinates(self.nx_resample, self.ny_resample)
        self.spectrum_45_degrees_centroid = EnergySpectrum(scale="auto", peak=511.0, max_energy=1500) 
        for index in range(len(self.data_45_degrees)):
            [xrec,yrec] = self.coordinates_45_degrees_centroid[index]
            energyrec = self.energy_45_degrees_centroid[index]
            self.histogram_45_degrees_centroid.add_data([xrec,yrec])
            self.spectrum_45_degrees_centroid.add_data(energyrec)

    def reconstruct_45_degrees_2D_MAP(self): 
        self.coordinates_45_degrees_2D_MAP = []
        self.energy_45_degrees_2D_MAP = []
        reconstructor = ReconstructorMAP(self.forward_model_2D)  
        for index in range(len(self.data_45_degrees)): 
            self.print_percentage(index,len(self.data_45_degrees))
            data = self.data_45_degrees[index]
            [xrec,yrec,energyrec,logposterior] = reconstructor.reconstruct(data, batch_size=BATCH_SIZE) 
            self.coordinates_45_degrees_2D_MAP.append([xrec,yrec])
            self.energy_45_degrees_2D_MAP.append(energyrec)
        return self.coordinates_45_degrees_2D_MAP
    
    def make_histograms_45_degrees_2D_MAP(self): 
        self.histogram_45_degrees_2D_MAP = HistogramCoordinates(self.nx_resample, self.ny_resample)
        self.spectrum_45_degrees_2D_MAP = EnergySpectrum(scale="auto", peak=511.0, max_energy=1500) 
        for index in range(len(self.data_45_degrees)): 
            [xrec,yrec] = self.coordinates_45_degrees_2D_MAP[index]
            energyrec = self.energy_45_degrees_2D_MAP[index]
            self.histogram_45_degrees_2D_MAP.add_data([xrec,yrec])
            self.spectrum_45_degrees_2D_MAP.add_data(energyrec)

    def reconstruct_45_degrees_3D_MAP_DE(self): 
        self.coordinates_45_degrees_3D_MAP_DE = []
        self.energy_45_degrees_3D_MAP_DE = []
        reconstructor = ReconstructorMAP(self.forward_model_DE)  
        reconstructor.set_prior(self.prior)
        for index in range(len(self.data_45_degrees)): 
            self.print_percentage(index,len(self.data_45_degrees))
            data = self.data_45_degrees[index]
            [xrec,yrec,zrec,energyrec,logposterior] = reconstructor.reconstruct(data, batch_size=BATCH_SIZE) 
            self.coordinates_45_degrees_3D_MAP_DE.append([xrec,yrec,zrec])
            self.energy_45_degrees_3D_MAP_DE.append(energyrec)
        return self.coordinates_45_degrees_3D_MAP_DE

    def make_histograms_45_degrees_3D_MAP_DE(self):
        self.histogram_45_degrees_3D_MAP_DE = HistogramCoordinates(self.nx_resample, self.ny_resample, self.nz_resample)
        self.spectrum_45_degrees_3D_MAP_DE = EnergySpectrum(scale="auto", peak=511.0, max_energy=1500) 
        for index in range(len(self.data_45_degrees)): 
            [xrec,yrec,zrec] = self.coordinates_45_degrees_3D_MAP_DE[index]
            energyrec = self.energy_45_degrees_3D_MAP_DE[index]
            self.histogram_45_degrees_3D_MAP_DE.add_data([xrec,yrec,zrec])
            self.spectrum_45_degrees_3D_MAP_DE.add_data(energyrec)

    def reconstruct_45_degrees_3D_MAP_MLEEM(self): 
        self.coordinates_45_degrees_3D_MAP_MLEEM = []
        self.energy_45_degrees_3D_MAP_MLEEM = []
        reconstructor = ReconstructorMAP(self.forward_model_MLEEM)  
        reconstructor.set_prior(self.prior)
        for index in range(len(self.data_45_degrees)): 
            self.print_percentage(index,len(self.data_45_degrees))
            data = self.data_45_degrees[index]
            [xrec,yrec,zrec,energyrec,logposterior] = reconstructor.reconstruct(data, batch_size=BATCH_SIZE) 
            self.coordinates_45_degrees_3D_MAP_MLEEM.append([xrec,yrec,zrec])
            self.energy_45_degrees_3D_MAP_MLEEM.append(energyrec)
        return self.coordinates_45_degrees_3D_MAP_MLEEM

    def make_histograms_45_degrees_3D_MAP_MLEEM(self):
        self.histogram_45_degrees_3D_MAP_MLEEM = HistogramCoordinates(self.nx_resample, self.ny_resample, self.nz_resample)
        self.spectrum_45_degrees_3D_MAP_MLEEM = EnergySpectrum(scale="auto", peak=511.0, max_energy=1500) 
        for index in range(len(self.data_45_degrees)): 
                [xrec,yrec,zrec] = self.coordinates_45_degrees_3D_MAP_MLEEM[index]
                energyrec = self.energy_45_degrees_3D_MAP_MLEEM[index]
                self.histogram_45_degrees_3D_MAP_MLEEM.add_data([xrec,yrec,zrec])
                self.spectrum_45_degrees_3D_MAP_MLEEM.add_data(energyrec)


    def compute_bias_and_variance_45_degrees(self): 
        h = self.histogram_45_degrees_3D_MAP_DE.histogram
        z = scipy.ndimage.interpolation.zoom(h,zoom=((50.0/self.ny_resample)/(15.0/self.nz_resample),(50.0/self.ny_resample)/(15.0/self.nz_resample), 1.0 ))
        r = scipy.ndimage.interpolation.rotate(z, angle=-45.0, axes=(1,2))
        v1 = r[:,:,np.int32((35.0*self.nx_resample)/self.nx):np.int32((45.0*self.nx_resample)/self.nx)].mean(2)[np.int32((23.5*self.nx_resample)/self.nx),:]
        x1 = np.linspace(0,len(v1)*25.0/z.shape[0],len(v1))
        v2 = r[:,:,np.int32((35.0*self.nx_resample)/self.nx):np.int32((45.0*self.nx_resample)/self.nx)].mean(2)[:,np.int32((27.0*self.nx_resample)/self.nx)]
        x2 = np.linspace(0,len(v2)*25.0/z.shape[0],len(v2))
        vn1 = v1/v1.sum()
        vn2 = v2/v2.sum()
        print "Beam std Y-Z: ",np.sqrt(((x1-(x1*vn1).sum())**2*vn1).sum())
        print "Beam std X:   ",np.sqrt(((x2-(x2*vn2).sum())**2*vn2).sum())
        pl.figure()
        pl.subplot(1,2,1)
        pl.imshow(r[:,:,np.int32((35.0*self.nx_resample)/self.nx):np.int32((45.0*self.nx_resample)/self.nx)].mean(2),cmap='hot')
        pl.subplot(1,2,2)
        pl.imshow(r.sum(0),cmap='hot')

    def visualize_results_45_degrees(self, vmax=100): 
        # Visualize histograms of the locations of interaction: 
        if self.coordinates_45_degrees_3D_MAP_DE is not None: 
            self.histogram_45_degrees_3D_MAP_DE.show(axis=0, index=None, vmax=vmax)
        if self.coordinates_45_degrees_3D_MAP_MLEEM is not None: 
            self.histogram_45_degrees_3D_MAP_MLEEM.show(axis=0, index=None, vmax=vmax)
        
        # Visualize histograms of the energy: 
        fig = pl.figure()
        if self.spectrum_45_degrees_centroid is not None: 
            self.spectrum_45_degrees_centroid.show(fig=fig.number)
        if self.spectrum_45_degrees_2D_MAP is not None: 
            self.spectrum_45_degrees_2D_MAP.show(fig=fig.number)
        if self.spectrum_45_degrees_3D_MAP_DE is not None: 
            self.spectrum_45_degrees_3D_MAP_DE.show(fig=fig.number)
        if self.spectrum_45_degrees_3D_MAP_MLEEM is not None: 
            self.spectrum_45_degrees_3D_MAP_MLEEM.show(fig=fig.number)


    def run(self): 
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

        print "-Upsampling the forward models",[self.nx_resample, self.ny_resample, self.nz_resample],".."
        self.resample_forward_model_2D([self.nx_resample, self.ny_resample]) 
        self.resample_forward_model_DE([self.nx_resample, self.ny_resample, self.nz_resample]) 
        self.resample_forward_model_MLEEM([self.nx_resample, self.ny_resample, self.nz_resample]) 
        prior_model = BeerLambert(self.scintillator_attenuation_coefficient, self.nz_resample, self.scintillator_thickness/self.nz)
        self.prior = prior_model.get_probability_mass_function()

        print "-Loading 2D test grid" 
        #self.load_test_grid([0,8,16,24], [0,8,16,24,32,40,48])
        self.load_test_grid([2,13,24], [2,13,24,35,46])
        print "-Reconstruction using centroid algorithm"
        self.reconstruct_grid_centroid(shiftx=2.9, shifty=2.5, scale=8.0*self.nx_resample/26.0)
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
        print "-Reconstruction using 3D maximum-a-posteriori (DepthEmbedding + MLEE model)"
        if self.load_reconstructions_grid_3D_MAP_MLEEM() is None: 
            self.reconstruct_grid_3D_MAP_MLEEM()
            self.save_reconstructions_grid_3D_MAP_MLEEM()
        self.filter_events_3D_MAP_MLEEM(fraction=0.7)
        self.make_histograms_grid_3D_MAP_MLEEM()
        print "Computing Bias and Variance"
        self.compute_bias_and_variance_grid() 
        print "Visualizing results .."
        self.visualize_results_grid() 

        print "-Loading 45 degrees test data"
        self.load_test_45_degrees([50,]) 
        print "-Reconstruction using centroid algorithm"
        self.reconstruct_45_degrees_centroid(shiftx=2.9, shifty=2.5, scale=8.0*self.nx_resample/26.0)
        self.make_histograms_45_degrees_centroid()
        print "-Reconstruction using 2D maximum-a-posteriori"
        if self.load_reconstructions_45_degrees_2D_MAP() is None: 
            self.reconstruct_45_degrees_2D_MAP()
            self.save_reconstructions_45_degrees_2D_MAP()
        self.make_histograms_45_degrees_2D_MAP()
        print "-Reconstruction using 3D maximum-a-posteriori (DepthEmbedding model)"
        if self.load_reconstructions_45_degrees_3D_MAP_DE() is None: 
            self.reconstruct_45_degrees_3D_MAP_DE()
            self.save_reconstructions_45_degrees_3D_MAP_DE()
        self.make_histograms_45_degrees_3D_MAP_DE()
        print "-Reconstruction using 3D maximum-a-posteriori (DepthEmbedding + MLEE model)"
        if self.load_reconstructions_45_degrees_3D_MAP_MLEEM() is None: 
            self.reconstruct_45_degrees_3D_MAP_MLEEM()
            self.save_reconstructions_45_degrees_3D_MAP_MLEEM()
        self.make_histograms_45_degrees_3D_MAP_MLEEM()
        print "Computing Bias and Variance"
        self.compute_bias_and_variance_45_degrees() 
        print "Visualizing results"
        self.visualize_results_45_degrees() 
        print "TestCmice Done"

    def get_data_grid(self, x,y):
        return get_data_cmice_grid(x,y, path=self.input_data_path)

    def get_data_45_degrees(self, index): 
        return get_data_cmice_45_degrees(index, path=self.input_data_path)

    def print_percentage(self,value,max_value):
        print "Progress: %d%%"%np.int32((100*(value+1))/(max_value))






if __name__ == "__main__": 
    print "--- DepthEmbedding ---"
    print "-- Calibration of cMiCE PET camera with 15mm-thick crystal, without resampling of the forward model.."
    cmice = Cmice_15mm(nx_resample=26, ny_resample=49, nz_resample=20, n_neighbours=12, n_training=2500, nz=20)
    cmice.run() 
    
    print "-- Calibration of cMiCE PET camera with 15mm-thick crystal, with resampling of the forward model.."
    cmice = Cmice_15mm(nx_resample=52, ny_resample=98, nz_resample=40, n_neighbours=12, n_training=2500, nz=20)
    cmice.run() 

    print "-- Done"

