

from depth_embedding import ReconstructorMAP, ReconstructorCentroid, get_cumulative_prior
from depth_embedding import BeerLambert, EnergySpectrum, HistogramCoordinates, LikelihoodFilter
from depth_embedding import ModelInterpolator, Model2D, ModelDepthEmbedding, ModelMLEEM, StatsCoordinates, model_error 

import scipy
import scipy.io
from numpy import zeros, ones, sort, unravel_index, repeat, sum, where, squeeze, fliplr, flipud
from numpy import log, tile, float32, argsort, int32, histogram, linspace, round, exp, convolve, sqrt, mgrid
import scipy.ndimage.filters 
import numpy as np 
import copy 
import os 
import pylab as pl


BATCH_SIZE = 128  # Number of events reconstructed in parallel (i.e. in a single batch). 
                  # If set to 0, all events are reconstructed at once: faster but may starve memory when 
                  # using large reconstruction grids (e.g. when up-sampling the forward model). 



def load_simulation_data_single(x, y, nz=32, path='./input_data_cmice_sim_15mm/SimSET_data/'):
    data = scipy.io.loadmat(path+'ROW_%s_COL_%s.mat'%(str(x+1).zfill(2),str(y+1).zfill(2)))
    model = np.zeros([nz,8,8])
    for z in range(nz): 
        dataz = data['data_unscattered'][:,data['discrete_depth'][0,:]==z] 
        model[z,:,:] = dataz.mean(1).reshape([8,8]) 
    model_flipped = np.zeros(model.shape)
    for z in range(nz):
        model_flipped[z,:,:] = model[nz-1-z,:,:]
    model = model.reshape([model.shape[0],model.shape[1],model.shape[2],64])
    return model_flipped


def load_simulation_data(path='./input_data_cmice_sim_15mm/SimSET_data/', nx=49, ny=49, nz=32): 
    model = zeros([nx,ny,nz,8,8]) 
    for x in range(nx): 
        for y in range(ny): 
            m = load_simulation_data_single(x,y, nz, path)
            model[x,y,:,:,:] = m
    model = model.reshape([model.shape[0],model.shape[1],model.shape[2],64])
    return model


def simulation_beam(forward_model, N=1, thickness=1.5, attenuation_coeff=0.83, noise_type='poisson', noise_sigma=0.1): 
    data  = np.zeros([N,64])

    nz = forward_model.shape[0]
    P = BeerLambert(attenuation_coeff, nz, thickness/nz)
    depth = P.sample(N)
    
    for i in range(N): 
        if noise_type is 'poisson': 
            data[i,:] = np.random.poisson( forward_model[depth[i],:].reshape([64,]) )
        else: 
            data[i,:] = np.random.normal( forward_model[depth[i],:].reshape([64,]), noise_sigma )
    return data, depth




def test(): 
    n_neighbours = 12
    nz           = 32  
    N_simu       = 50000 
    N_train      = 5000

    # Load forward model from Monte Carlo simulation 
    model = load_simulation_data_single(25,25)
    
    # Simulate a single beam measurement 
    scale = 0.1
    beam_data, beam_depth = simulation_beam(forward_model=scale*model, N=N_simu)
    
    # Estimate local forward model using DepthEmbedding and visualize the manifold  
    prior = BeerLambert(0.83, 32, 1.5/nz).get_probability_mass_function() 

    model_estimator = ModelDepthEmbedding(nz=nz, n_neighbours=n_neighbours) 
    model_estimator.set_calibration_data(beam_data[0:N_train]) 
    model_estimator.set_depth_prior(prior) 
    model_e = model_estimator.estimate_forward_model(allow_negative_values=True) 
    
    model_estimator.visualize_manifold() 
    
    pl.figure()
    pl.subplot(231); pl.imshow(model[0,0,0,:].reshape([8,8]),interpolation='nearest')
    pl.subplot(232); pl.imshow(model[0,0,15,:].reshape([8,8]),interpolation='nearest')
    pl.subplot(233); pl.imshow(model[0,0,31,:].reshape([8,8]),interpolation='nearest')
    pl.subplot(234); pl.imshow(model_e[0,:].reshape([8,8]),interpolation='nearest')
    pl.subplot(235); pl.imshow(model_e[15,:].reshape([8,8]),interpolation='nearest')
    pl.subplot(236); pl.imshow(model_e[31,:].reshape([8,8]),interpolation='nearest')




class CmiceSimulation_15mm(): 
    """Estimation of the forward model of the cMiCe PET camera and reconstruction using various 2D and 3D methods - simulated cMiCE data. """
    def __init__(self, n_neighbours=12, n_training=2700, n_testing=10000, nz=24, nx_resample=49, ny_resample=49, nz_resample=24): 
        self.nx = 49
        self.ny = 49 
        self.nx_resample = nx_resample
        self.ny_resample = ny_resample
        self.nz_resample = nz_resample
        self.n_detectors = 64 
        self.scintillator_attenuation_coefficient = 0.83  #[cm-1]
        self.scintillator_thickness = 1.5                 #[cm]
        self.input_data_path = "./input_data_cmice_sim_15mm/"
        self.results_path    = "./results_cmice_sim_15mm/"
        self.nz = nz
        self.n_neighbours = n_neighbours
        self.n_training   = n_training 
        self.n_testing    = n_testing
        
        self._set_depth_prior()
        self.forward_model_2D = None 
        self.forward_model_DE = None 
        self.forward_model_MLEEM = None 
        self.coordinates_centroid = None 
        self.coordinates_2D_MAP = None 
        self.coordinates_3D_MAP_DE = None 
        self.coordinates_3D_MAP_MLEEM = None 
        self.histogram_centroid = None
        self.histogram_2D_MAP = None
        self.histogram_3D_MAP_DE = None
        self.histogram_3D_MAP_MLEEM = None
        self.spectrum_centroid = None
        self.spectrum_2D_MAP = None
        self.spectrum_3D_MAP_DE = None
        self.spectrum_3D_MAP_MLEEM = None
        


    def _set_depth_prior(self): 
        prior_model = BeerLambert(alpha_inv_cm=self.scintillator_attenuation_coefficient, n_bins=self.nz, bin_size_cm=self.scintillator_thickness/self.nz)
        self.prior = prior_model.get_probability_mass_function()

    def load_forward_model_2D(self, filename='/model_cmice_sim_2D.mat'): 
        filename = self.results_path+filename
        try: 
            model = scipy.io.loadmat(filename)['model_cmice_sim_2D']
        except: 
            return None
        print "-Found 2D forward model file: %s - not recomputing it."%filename
        self.forward_model_2D = model
        return model

    def load_forward_model_DE(self, filename='/model_cmice_sim_DE.mat'): 
        filename = self.results_path+filename
        try: 
            model = scipy.io.loadmat(filename)['model_cmice_sim_DE']
        except: 
            return None
        print "-Found 3D DepthEmbedding forward model file: %s - not recomputing it."%filename
        self.forward_model_DE = model
        return model

    def load_forward_model_MLEEM(self, filename='/model_cmice_sim_MLEEM.mat'): 
        filename = self.results_path+filename
        try: 
            model = scipy.io.loadmat(filename)['model_cmice_sim_MLEEM']
        except: 
            return None
        print "-Found 3D DepthEmbedding+MLEEM forward model file: %s - not recomputing it."%filename
        self.forward_model_MLEEM = model
        return model


    def save_forward_model_2D(self, filename='/model_cmice_sim_2D.mat'):
        filename = self.results_path+filename 
        scipy.io.savemat(filename, {'model_cmice_sim_2D':self.forward_model_2D})

    def save_forward_model_DE(self, filename='/model_cmice_sim_DE.mat'):
        filename = self.results_path+filename 
        scipy.io.savemat(filename, {'model_cmice_sim_DE':self.forward_model_DE})
 
    def save_forward_model_MLEEM(self, filename='/model_cmice_sim_MLEEM.mat'): 
        filename = self.results_path+filename
        scipy.io.savemat(filename, {'model_cmice_sim_MLEEM':self.forward_model_MLEEM})


    def load_reconstructions_grid_2D_MAP(self, filename='/coordinates_sim_2D_MAP'): 
        filename = self.results_path+filename
        filename = filename + '_%d_%d_%d.mat'%(self.nx_resample, self.ny_resample, self.nz_resample)
        try: 
            c = scipy.io.loadmat(filename)['coordinates_sim_2D_MAP']
            e = scipy.io.loadmat(filename)['energy_sim_2D_MAP']
        except: 
            return None
        print "-Found 2D MAP reconstructed coordinates file: %s - not recomputing them."%filename
        coords = []
        energy = []
        ngridx = c.shape[0]
        ngridy = c.shape[1]
        for ix in range(ngridx): 
            coords_row = []
            energy_row = []
            for iy in range(ngridy):
                coords_row.append(c[ix,iy,:])
                energy_row.append(e[ix,iy,:])
            coords.append(coords_row)
            energy.append(energy_row)
        self.coordinates_2D_MAP = coords
        self.energy_2D_MAP = energy
        return coords, energy

    def load_reconstructions_grid_3D_MAP_DE(self, filename='/coordinates_sim_3D_MAP_DE'): 
        filename = self.results_path+filename
        filename = filename + '_%d_%d_%d.mat'%(self.nx_resample, self.ny_resample, self.nz_resample)
        try: 
            c = scipy.io.loadmat(filename)['coordinates_sim_3D_MAP_DE']
            e = scipy.io.loadmat(filename)['energy_sim_3D_MAP_DE']
        except: 
            return None
        print "-Found 3D MAP DepthEmbedding reconstructed coordinates file: %s - not recomputing them."%filename
        coords = []
        energy = []
        ngridx = c.shape[0]
        ngridy = c.shape[1]
        for ix in range(ngridx): 
            coords_row = []
            energy_row = []
            for iy in range(ngridy):
                coords_row.append(c[ix,iy,:,:])
                energy_row.append(e[ix,iy,:])
            coords.append(coords_row)
            energy.append(energy_row)
        self.coordinates_3D_MAP_DE = coords
        self.energy_3D_MAP_DE = energy
        return coords, energy

    def load_reconstructions_grid_3D_MAP_true(self, filename='/coordinates_sim_3D_MAP_true'): 
        filename = self.results_path+filename
        filename = filename + '_%d_%d_%d.mat'%(self.nx_resample, self.ny_resample, self.nz_resample)
        try: 
            c = scipy.io.loadmat(filename)['coordinates_sim_3D_MAP_true']
            e = scipy.io.loadmat(filename)['energy_sim_3D_MAP_true']
        except: 
            return None
        print "-Found 3D MAP DepthEmbedding reconstructed coordinates file: %s - not recomputing them."%filename
        coords = []
        energy = []
        ngridx = c.shape[0]
        ngridy = c.shape[1]
        for ix in range(ngridx): 
            coords_row = []
            energy_row = []
            for iy in range(ngridy):
                coords_row.append(c[ix,iy,:,:])
                energy_row.append(e[ix,iy,:])
            coords.append(coords_row)
            energy.append(energy_row)
        self.coordinates_3D_MAP_true = coords
        self.energy_3D_MAP_true = energy
        return coords, energy

    def load_reconstructions_grid_3D_MAP_MLEEM(self, filename='/coordinates_sim_3D_MAP_MLEEM'): 
        filename = self.results_path+filename
        filename = filename + '_%d_%d_%d.mat'%(self.nx_resample, self.ny_resample, self.nz_resample)
        try: 
            c = scipy.io.loadmat(filename)['coordinates_sim_3D_MAP_MLEEM']
            e = scipy.io.loadmat(filename)['energy_sim_3D_MAP_MLEEM']
        except: 
            return None
        print "-Found 3D MAP DepthEmbedding+MLEEM reconstructed coordinates file: %s - not recomputing them."%filename
        coords = []
        energy = []
        ngridx = c.shape[0]
        ngridy = c.shape[1]
        for ix in range(ngridx): 
            coords_row = []
            energy_row = []
            for iy in range(ngridy):
                coords_row.append(c[ix,iy,:,:])
                energy_row.append(e[ix,iy,:])
            coords.append(coords_row)
            energy.append(energy_row)
        self.coordinates_3D_MAP_MLEEM = coords
        self.energy_3D_MAP_MLEEM = energy
        return coords, energy

    def save_reconstructions_grid_2D_MAP(self, filename='/coordinates_sim_2D_MAP'): 
        filename = self.results_path+filename
        filename = filename + '_%d_%d_%d.mat'%(self.nx_resample, self.ny_resample, self.nz_resample)
        scipy.io.savemat(filename, {'coordinates_sim_2D_MAP':self.coordinates_2D_MAP,'energy_sim_2D_MAP':self.energy_2D_MAP})

    def save_reconstructions_grid_3D_MAP_DE(self, filename='/coordinates_sim_3D_MAP_DE'): 
        filename = self.results_path+filename
        filename = filename + '_%d_%d_%d.mat'%(self.nx_resample, self.ny_resample, self.nz_resample)
        scipy.io.savemat(filename, {'coordinates_sim_3D_MAP_DE':self.coordinates_3D_MAP_DE,'energy_sim_3D_MAP_DE':self.energy_3D_MAP_DE})

    def save_reconstructions_grid_3D_MAP_true(self, filename='/coordinates_sim_3D_MAP_true'): 
        filename = self.results_path+filename
        filename = filename + '_%d_%d_%d.mat'%(self.nx_resample, self.ny_resample, self.nz_resample)
        scipy.io.savemat(filename, {'coordinates_sim_3D_MAP_true':self.coordinates_3D_MAP_true,'energy_sim_3D_MAP_true':self.energy_3D_MAP_true})

    def save_reconstructions_grid_3D_MAP_MLEEM(self, filename='/coordinates_sim_3D_MAP_MLEEM'): 
        filename = self.results_path+filename
        filename = filename + '_%d_%d_%d.mat'%(self.nx_resample, self.ny_resample, self.nz_resample)
        scipy.io.savemat(filename, {'coordinates_sim_3D_MAP_MLEEM':self.coordinates_3D_MAP_MLEEM,'energy_sim_3D_MAP_MLEEM':self.energy_3D_MAP_MLEEM})

    def estimate_forward_model_2D(self): 
        model = zeros([self.nx, self.ny, self.n_detectors])
        for ix in range(self.nx): 
            #self.print_percentage(ix,self.nx)
            for iy in range(self.ny): 
                self.print_percentage(iy+ix*self.ny,self.ny*self.nx)
                calibration_data, depth = self.get_data_grid(ix,iy, N=self.n_training)
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
                calibration_data, depth = self.get_data_grid(ix,iy, N=self.n_training)
                #filter = LikelihoodFilter(self.forward_model_2D)
                #calibration_data_filtered, rejected = filter.filter(calibration_data, ix,iy, method="near_coordinates", points=self.n_training) 
                calibration_data_filtered = calibration_data[0:self.n_training,:]
                print "number of points for location %d,%d:"%(ix,iy),calibration_data_filtered.shape[0]
                model_estimator = ModelDepthEmbedding(nz=self.nz, n_neighbours=self.n_neighbours)
                model_estimator.set_calibration_data(calibration_data_filtered)
                model_estimator.set_depth_prior(self.prior)
                try: 
                    model[ix,iy,:,:] = model_estimator.estimate_forward_model(unit_norm=True, zero_mean=True) 
                except(ValueError):
                    print "Problem with dimensionality reduction, zero mean desabled."
                    try: 
                        model[ix,iy,:,:] = model_estimator.estimate_forward_model(unit_norm=True, zero_mean=False) 
                    except(ValueError):
                        print "Problem with dimensionality reduction, unit norm desabled."
                        try: 
                            model[ix,iy,:,:] = model_estimator.estimate_forward_model(unit_norm=False, zero_mean=False)
                        except(ValueError):
                            raise    
        self.forward_model_DE = model 
        return self.forward_model_DE

    def estimate_forward_model_MLEEM(self, sigma_smoothness=0.0, iterations_smoothness=5): 
        model = zeros([self.nx, self.ny, self.nz, self.n_detectors])
        for ix in range(self.nx): 
            for iy in range(self.ny): 
                self.print_percentage(iy+ix*self.ny,self.ny*self.nx)
                calibration_data, depth = self.get_data_grid(ix,iy, N=self.n_training)
                #filter = LikelihoodFilter(self.forward_model_2D)
                #calibration_data_filtered, rejected = filter.filter(calibration_data, ix,iy, points=self.n_training) 
                calibration_data_filtered = calibration_data[0:self.n_training,:]
                model_estimator = ModelMLEEM(initial_model=self.forward_model_DE[ix,iy,:,:])
                model_estimator.set_calibration_data(calibration_data_filtered) 
                model_estimator.set_depth_prior(self.prior) 
                model[ix,iy,:,:] = model_estimator.estimate_forward_model() 
        self.forward_model_MLEEM = model
        if sigma_smoothness > 1e-9: 
            print "MLE-EM smoothness refinement .."
            for ii in range(iterations_smoothness): 
                model_smoothed = scipy.ndimage.filters.gaussian_filter(model, sigma=sigma_smoothness, mode='reflect') 
                for ix in range(self.nx): 
                    for iy in range(self.ny): 
                        self.print_percentage(iy+ix*self.ny+ii*self.ny*self.nx,iterations_smoothness*self.ny*self.nx)
                        calibration_data, depth = self.get_data_grid(ix,iy, N=self.n_training)
                        #filter = LikelihoodFilter(self.forward_model_2D)
                        #calibration_data_filtered, rejected = filter.filter(calibration_data, ix,iy, points=self.n_training) 
                        calibration_data_filtered = calibration_data[0:self.n_training,:]
                        model_estimator = ModelMLEEM(initial_model=model_smoothed[ix,iy,:,:])
                        model_estimator.set_calibration_data(calibration_data_filtered) 
                        model_estimator.set_depth_prior(self.prior) 
                        model[ix,iy,:,:] = model_estimator.estimate_forward_model() 
            self.forward_model_MLEEM = model   
        return self.forward_model_MLEEM


    def resample_forward_model_2D(self, grid_size, interpolation='cubic', noise_sigma=0.1): 
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

    def resample_forward_model_DE(self, grid_size, interpolation='cubic', noise_sigma=0.1): 
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

    def resample_forward_model_true(self, grid_size, interpolation='cubic', noise_sigma=0.1): 
        if self.nx == self.nx_resample and self.ny == self.ny_resample and self.nz == self.nz_resample: 
            print "Not resampling MLEEM model."
            return self.forward_model_MLEEM
        interpolator = ModelInterpolator()
        forward_model = interpolator.resample(self.forward_model_true, grid_size, interpolation=interpolation)
        if noise_sigma > 1e-9:
            # A small amount of noise in the forward model effectively removes interpolation artefacts 
            forward_model = np.random.normal(forward_model, noise_sigma)
            forward_model[forward_model<=0.0]=0.0
        self.forward_model_true = forward_model
        return forward_model

    def resample_forward_model_MLEEM(self, grid_size, interpolation='cubic', noise_sigma=0.1): 
        interpolator = ModelInterpolator()
        forward_model = interpolator.resample(self.forward_model_MLEEM, grid_size, interpolation=interpolation)
        if noise_sigma > 1e-9:
            # A small amount of noise in the forward model effectively removes interpolation artefacts 
            forward_model = np.random.normal(forward_model, noise_sigma)
            forward_model[forward_model<=0.0]=0.0
        self.forward_model_MLEEM = forward_model
        return forward_model

    def load_test_grid(self, x_locations, y_locations): 
        self.grid       = []
        self.grid_depth = []
        for ix in x_locations: 
            data_row  = [] 
            depth_row = []
            for iy in y_locations: 
                data, depth = self.get_data_grid(ix,iy, N=self.n_testing)
                data_row.append(data)
                depth_row.append(depth)
            self.grid.append(data_row) 
            self.grid_depth.append(depth_row)
        self.grid_shape = (len(x_locations),len(y_locations))
        print self.grid_shape
        self.grid_locations = [tile(x_locations,(len(y_locations),1)).transpose(), tile(int32(y_locations).reshape((len(y_locations),1)),(1,len(x_locations))).transpose()]
        return self.grid
            
    def reconstruct_grid_centroid(self, shiftx=3.0, shifty=3.0, scale=7.0): 
        self.coordinates_centroid = [] 
        self.energy_centroid = []
        x_detectors = tile(linspace(0,9,8),(1,8))[0] - 4.5 
        y_detectors = repeat(linspace(0,9,8),8,axis=0) - 4.3
        reconstructor = ReconstructorCentroid(x_detectors=x_detectors, y_detectors=y_detectors, x_max=self.nx-1, y_max=self.ny-1, shiftx=shiftx, shifty=shifty, scale=scale, exponent=1.0)  
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
        self.histogram_centroid = HistogramCoordinates(self.nx_resample, self.ny_resample)
        self.spectrum_centroid = EnergySpectrum(scale="auto", peak=511.0, max_energy=1500) 
        for ix in range(self.grid_shape[0]): 
            for iy in range(self.grid_shape[1]):
                [xrec,yrec] = self.coordinates_centroid[ix][iy]
                xrec = xrec * self.nx_resample / self.nx
                yrec = yrec * self.ny_resample / self.ny
                energyrec = self.energy_centroid[ix][iy]
                self.histogram_centroid.add_data([xrec,yrec])
                self.spectrum_centroid.add_data(energyrec)

    def reconstruct_grid_2D_MAP(self): 
        self.coordinates_2D_MAP = []
        self.energy_2D_MAP = []
        reconstructor = ReconstructorMAP(self.forward_model_2D)  
        for ix in range(self.grid_shape[0]): 
            #self.print_percentage(ix,self.grid_shape[0])
            row_coordinates = []
            row_energy = []
            for iy in range(self.grid_shape[1]):
                self.print_percentage(iy+ix*self.grid_shape[1],self.grid_shape[0]*self.grid_shape[1])
                data = self.grid[ix][iy]
                [xrec,yrec,energyrec,logposterior] = reconstructor.reconstruct(data, batch_size=BATCH_SIZE) 
                row_coordinates.append([xrec,yrec])
                row_energy.append(energyrec)
            self.coordinates_2D_MAP.append(row_coordinates)
            self.energy_2D_MAP.append(row_energy)
        return self.coordinates_2D_MAP
    
    def make_histograms_grid_2D_MAP(self): 
        self.histogram_2D_MAP = HistogramCoordinates(self.nx_resample, self.ny_resample)
        self.spectrum_2D_MAP = EnergySpectrum(scale="auto", peak=511.0, max_energy=1500) 
        for ix in range(self.grid_shape[0]): 
            for iy in range(self.grid_shape[1]):
                [xrec,yrec] = self.coordinates_2D_MAP[ix][iy]
                energyrec = self.energy_2D_MAP[ix][iy]
                self.histogram_2D_MAP.add_data([xrec,yrec])
                self.spectrum_2D_MAP.add_data(energyrec)

    def reconstruct_grid_3D_MAP_DE(self): 
        self.coordinates_3D_MAP_DE = []
        self.energy_3D_MAP_DE = []
        reconstructor = ReconstructorMAP(self.forward_model_DE)  
        reconstructor.set_prior(self.prior)
        for ix in range(self.grid_shape[0]): 
            #self.print_percentage(ix,self.grid_shape[0])
            row_coordinates = []
            row_energy = []
            for iy in range(self.grid_shape[1]):
                self.print_percentage(iy+ix*self.grid_shape[1],self.grid_shape[0]*self.grid_shape[1])
                data = self.grid[ix][iy]
                [xrec,yrec,zrec,energyrec,logposterior] = reconstructor.reconstruct(data, batch_size=BATCH_SIZE) 
                row_coordinates.append([xrec,yrec,zrec])
                row_energy.append(energyrec)
            self.coordinates_3D_MAP_DE.append(row_coordinates)
            self.energy_3D_MAP_DE.append(row_energy)
        return self.coordinates_3D_MAP_DE

    def reconstruct_grid_3D_MAP_true(self): 
        self.coordinates_3D_MAP_true = []
        self.energy_3D_MAP_true = []
        reconstructor = ReconstructorMAP(self.forward_model_true)  
        reconstructor.set_prior(self.prior)
        for ix in range(self.grid_shape[0]): 
            #self.print_percentage(ix,self.grid_shape[0])
            row_coordinates = []
            row_energy = []
            for iy in range(self.grid_shape[1]):
                self.print_percentage(iy+ix*self.grid_shape[1],self.grid_shape[0]*self.grid_shape[1])
                data = self.grid[ix][iy]
                [xrec,yrec,zrec,energyrec,logposterior] = reconstructor.reconstruct(data, batch_size=BATCH_SIZE) 
                row_coordinates.append([xrec,yrec,zrec])
                row_energy.append(energyrec)
            self.coordinates_3D_MAP_true.append(row_coordinates)
            self.energy_3D_MAP_true.append(row_energy)
        return self.coordinates_3D_MAP_true

    def make_histograms_grid_3D_MAP_DE(self):
        self.histogram_3D_MAP_DE = HistogramCoordinates(self.nx_resample, self.ny_resample, self.nz_resample)
        self.spectrum_3D_MAP_DE = EnergySpectrum(scale="auto", peak=511.0, max_energy=1500) 
        for ix in range(self.grid_shape[0]): 
            for iy in range(self.grid_shape[1]):
                [xrec,yrec,zrec] = self.coordinates_3D_MAP_DE[ix][iy]
                energyrec = self.energy_3D_MAP_DE[ix][iy]
                self.histogram_3D_MAP_DE.add_data([xrec,yrec,zrec])
                self.spectrum_3D_MAP_DE.add_data(energyrec)

    def make_histograms_grid_3D_MAP_true(self):
        self.histogram_3D_MAP_true = HistogramCoordinates(self.nx_resample, self.ny_resample, self.nz_resample)
        self.spectrum_3D_MAP_true = EnergySpectrum(scale="auto", peak=511.0, max_energy=1500) 
        for ix in range(self.grid_shape[0]): 
            for iy in range(self.grid_shape[1]):
                [xrec,yrec,zrec] = self.coordinates_3D_MAP_true[ix][iy]
                energyrec = self.energy_3D_MAP_true[ix][iy]
                self.histogram_3D_MAP_true.add_data([xrec,yrec,zrec])
                self.spectrum_3D_MAP_true.add_data(energyrec)

    def reconstruct_grid_3D_MAP_MLEEM(self): 
        self.coordinates_3D_MAP_MLEEM = []
        self.energy_3D_MAP_MLEEM = []
        reconstructor = ReconstructorMAP(self.forward_model_MLEEM)  
        reconstructor.set_prior(self.prior)
        for ix in range(self.grid_shape[0]): 
            #self.print_percentage(ix,self.grid_shape[0])
            row_coordinates = []
            row_energy = []
            for iy in range(self.grid_shape[1]):
                self.print_percentage(iy+ix*self.grid_shape[1],self.grid_shape[0]*self.grid_shape[1])
                data = self.grid[ix][iy]
                [xrec,yrec,zrec,energyrec,logposterior] = reconstructor.reconstruct(data, batch_size=BATCH_SIZE) 
                row_coordinates.append([xrec,yrec,zrec])
                row_energy.append(energyrec)
            self.coordinates_3D_MAP_MLEEM.append(row_coordinates)
            self.energy_3D_MAP_MLEEM.append(row_energy)
        return self.coordinates_3D_MAP_MLEEM

    def make_histograms_grid_3D_MAP_MLEEM(self):
        self.histogram_3D_MAP_MLEEM = HistogramCoordinates(self.nx_resample, self.ny_resample, self.nz_resample)
        self.spectrum_3D_MAP_MLEEM = EnergySpectrum(scale="auto", peak=511.0, max_energy=1500) 
        for ix in range(self.grid_shape[0]): 
            for iy in range(self.grid_shape[1]):
                [xrec,yrec,zrec] = self.coordinates_3D_MAP_MLEEM[ix][iy]
                energyrec = self.energy_3D_MAP_MLEEM[ix][iy]
                self.histogram_3D_MAP_MLEEM.add_data([xrec,yrec,zrec])
                self.spectrum_3D_MAP_MLEEM.add_data(energyrec)


    def filter_events_2D_MAP(self, fraction=0.5): 
        self.coordinates_2D_MAP_filtered = []
        self.energy_2D_MAP_filtered = [] 
        self.likelihood_2D_MAP_filtered = []
        N = int32(len(self.coordinates_2D_MAP[ix][iy][0])*fraction)
        for ix in range(self.grid_shape[0]): 
            C = []
            E = []
            L = []
            for iy in range(self.grid_shape[1]):
                x = self.coordinates_2D_MAP[ix][iy][0]
                y = self.coordinates_2D_MAP[ix][iy][1]
                e = self.energy_2D_MAP[ix][iy]
                l = self.likelihood_2D_MAP[ix][iy]
                indexes = flipud(argsort(l)) 
                x = x[0][indexes][0:N]
                y = y[1][indexes][0:N]
                e = e[indexes][0:N]
                l = l[indexes][0:N]
                C.append(c)
                E.append(e)
                L.append(l)
            self.coordinates_2D_MAP_filtered.append(E) 
            self.energy_2D_MAP_filtered.append(E) 
            self.likelihood_2D_MAP_filtered.append(L)

    def filter_events_3D_MAP_DE(self, fraction=0.5): 
        self.coordinates_3D_MAP_DE_filtered = []
        self.energy_3D_MAP_DE_filtered = []
        self.likelihood_3D_MAP_DE_filtered = []
        N = int32(len(self.coordinates_3D_MAP_DE[ix][iy][0])*fraction)
        for ix in range(self.grid_shape[0]): 
            C = []
            E = []
            L = []
            for iy in range(self.grid_shape[1]):
                x = self.coordinates_3D_MAP_DE[ix][iy][0]
                y = self.coordinates_3D_MAP_DE[ix][iy][1]
                e = self.energy_3D_MAP_DE[ix][iy]
                l = self.likelihood_3D_MAP_DE[ix][iy]
                indexes = flipud(argsort(l)) 
                x = x[0][indexes][0:N]
                y = y[1][indexes][0:N]
                e = e[indexes][0:N]
                l = l[indexes][0:N]
                C.append(c)
                E.append(e)
                L.append(l)
            self.coordinates_3D_MAP_DE_filtered.append(E) 
            self.energy_3D_MAP_DE_filtered.append(E) 
            self.likelihood_3D_MAP_DE_filtered.append(L)
        
    def filter_events_3D_MAP_MLEEM(self, fraction=0.5): 
        self.coordinates_3D_MAP_MLEEM_filtered = []
        self.energy_3D_MAP_MLEEM_filtered = []
        self.likelihood_3D_MAP_MLEEM_filtered = []
        N = int32(len(self.coordinates_3D_MAP_MLEEM[ix][iy][0])*fraction)
        for ix in range(self.grid_shape[0]): 
            C = []
            E = []
            L = []
            for iy in range(self.grid_shape[1]):
                x = self.coordinates_3D_MAP_MLEEM[ix][iy][0]
                y = self.coordinates_3D_MAP_MLEEM[ix][iy][1]
                e = self.energy_3D_MAP_MLEEM[ix][iy]
                l = self.likelihood_3D_MAP_MLEEM[ix][iy]
                indexes = flipud(argsort(l)) 
                x = x[0][indexes][0:N]
                y = y[1][indexes][0:N]
                e = e[indexes][0:N]
                l = l[indexes][0:N]
                C.append(c)
                E.append(e)
                L.append(l)
            self.coordinates_3D_MAP_MLEEM_filtered.append(E) 
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
            stats_centroid = StatsCoordinates(self.coordinates_centroid, grid_locations, [(0,self.nx_resample),(0,self.ny_resample)], "Centroid", scale=(25.0/self.nx_resample, 50.0/self.ny_resample, 15.0/self.nz_resample))
            stats_centroid.print_summary()
        if self.coordinates_2D_MAP is not None: 
            stats_2D_MAP = StatsCoordinates(self.coordinates_2D_MAP, grid_locations, [(0,self.nx_resample),(0,self.ny_resample)], "2D MAP", scale=(25.0/self.nx_resample, 50.0/self.ny_resample, 15.0/self.nz_resample))
            stats_2D_MAP.print_summary()
        if self.coordinates_3D_MAP_DE is not None: 
            stats_3D_MAP_DE = StatsCoordinates(self.coordinates_3D_MAP_DE, grid_locations, [(0,self.nx_resample),(0,self.ny_resample),(2,self.nz_resample-2)], "3D MAP DepthEmbedding", scale=(25.0/self.nx_resample, 50.0/self.ny_resample, 15.0/self.nz_resample))
            stats_3D_MAP_DE.print_summary()
            #stats_3D_MAP_DE.plot_depth_density(6)
        if self.coordinates_3D_MAP_true is not None: 
            stats_3D_MAP_true = StatsCoordinates(self.coordinates_3D_MAP_true, grid_locations, [(0,self.nx_resample),(0,self.ny_resample),(2,self.nz_resample-2)], "True forward model", scale=(25.0/self.nx_resample, 50.0/self.ny_resample, 15.0/self.nz_resample))
            stats_3D_MAP_true.print_summary()
            #stats_3D_MAP_true.plot_depth_density(6)
        if self.coordinates_3D_MAP_MLEEM is not None: 
            stats_3D_MAP_MLEEM = StatsCoordinates(self.coordinates_3D_MAP_MLEEM, grid_locations, [(0,self.nx_resample),(0,self.ny_resample),(2,self.nz_resample-2)], "3D MAP DepthEmbedding+MLEEM", scale=(25.0/self.nx_resample, 50.0/self.ny_resample, 15.0/self.nz_resample))
            stats_3D_MAP_MLEEM.print_summary()
            #stats_3D_MAP_MLEEM.plot_depth_density(6)
            

    def visualize_results_grid(self, n_points=5000): 
        # Visualize manifold: 
        if n_points is None or n_points is 0.0: 
            n_points = self.n_training 
        ix = 15
        iy = 15
        Ix = ix*1.0 / self.nx * self.nx_resample
        Iy = iy*1.0 / self.ny * self.ny_resample
        calibration_data, depth = self.get_data_grid(ix,iy, N=self.n_training) 
        calibration_data_filtered = calibration_data[0:self.n_training,:]
        model_estimator = ModelDepthEmbedding(nz=self.nz, n_neighbours=self.n_neighbours)
        model_estimator.set_calibration_data(calibration_data_filtered)
        model_estimator.set_depth_prior(self.prior)
        model_estimator.estimate_forward_model() 
        model_estimator.visualize_manifold(nd=3)

        # Visualize histograms of the locations of interaction: 
        if self.coordinates_centroid is not None: 
            self.histogram_centroid.show()
        if self.coordinates_2D_MAP is not None: 
            self.histogram_2D_MAP.show()
        if self.coordinates_3D_MAP_DE is not None: 
            self.histogram_3D_MAP_DE.show()
        if self.coordinates_3D_MAP_true is not None: 
            self.histogram_3D_MAP_true.show()
        if self.coordinates_3D_MAP_MLEEM is not None: 
            self.histogram_3D_MAP_MLEEM.show()
        
        # Visualize histograms of the energy: 
        if self.spectrum_centroid is not None: 
            self.spectrum_centroid.show(5)
        if self.spectrum_2D_MAP is not None: 
            self.spectrum_2D_MAP.show(5)
        if self.spectrum_3D_MAP_DE is not None: 
            self.spectrum_3D_MAP_DE.show(5)
        if self.spectrum_3D_MAP_true is not None: 
            self.spectrum_3D_MAP_true.show(5)
        if self.spectrum_3D_MAP_MLEEM is not None: 
            self.spectrum_3D_MAP_MLEEM.show(5)

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


    def load_forward_model_true(self, filename='/model_cmice_sim_true.mat'):
        filename = self.results_path+filename
        try: 
            model = scipy.io.loadmat(filename)['model_cmice_sim_true']
        except: 
            return None
        model = model.reshape([model.shape[0],model.shape[1],model.shape[2],64])
        self.forward_model_true = model 
        return model 
        
    def load_forward_model_true_from_simulation(self, scale=0.25, depth_range=[4,28]): 
        model = scale * load_simulation_data(nx=49, ny=49, nz=32)
        model = model[:,:,range(depth_range[0], depth_range[1]),:,:]
        model = model.reshape([model.shape[0],model.shape[1],model.shape[2],64])
        self.forward_model_true = model
        return model
    
    def save_forward_model_true(self, filename='/model_cmice_sim_true.mat'): 
        filename = self.results_path+filename
        scipy.io.savemat(filename, {'model_cmice_sim_true':self.forward_model_true})
        
    def run(self): 
        print "-Loading ground truth forward model .."
        if self.load_forward_model_true() is None: 
            self.load_forward_model_true_from_simulation()
            self.save_forward_model_true()
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
    
        print "Model error DepthEmbedding:       %3.2f%%"%model_error(self.forward_model_true,self.forward_model_DE)
        print "Model error DepthEmbedding+MLEEM: %3.2f%%"%model_error(self.forward_model_true,self.forward_model_MLEEM)

        print "-Loading 2D test grid"
        #self.load_test_grid([0,8,16,24], [0,8,16,24,32,40,48])
        self.load_test_grid([2,13,24,35,46], [2,13,24,35,46])
        #self.load_test_grid([16,24,], [32,48,])

        print "-Upsampling the forward models",[self.nx_resample, self.ny_resample, self.nz_resample],".."
        self.resample_forward_model_2D([self.nx_resample, self.ny_resample]) 
        self.resample_forward_model_DE([self.nx_resample, self.ny_resample, self.nz_resample]) 
        self.resample_forward_model_true([self.nx_resample, self.ny_resample, self.nz_resample]) 
        self.resample_forward_model_MLEEM([self.nx_resample, self.ny_resample, self.nz_resample]) 
        prior_model = BeerLambert(self.scintillator_attenuation_coefficient, self.nz_resample, self.scintillator_thickness/self.nz)
        self.prior = prior_model.get_probability_mass_function()

        print "-Reconstruction using centroid algorithm"
        self.reconstruct_grid_centroid(shiftx=3.6,shifty=3.5, scale=6.5 * self.nx_resample/49.0)
        self.make_histograms_grid_centroid()
        print "-Reconstruction using 2D maximum-a-posteriori"
        if self.load_reconstructions_grid_2D_MAP() is None: 
            self.reconstruct_grid_2D_MAP()
            self.save_reconstructions_grid_2D_MAP()
        self.make_histograms_grid_2D_MAP()
        print "-Reconstruction using 3D maximum-a-posteriori (true model)"
        if self.load_reconstructions_grid_3D_MAP_true() is None: 
            self.reconstruct_grid_3D_MAP_true()
            self.save_reconstructions_grid_3D_MAP_true()
        self.make_histograms_grid_3D_MAP_true()
        print "-Reconstruction using 3D maximum-a-posteriori (DepthEmbedding model)"
        if self.load_reconstructions_grid_3D_MAP_DE() is None: 
            self.reconstruct_grid_3D_MAP_DE()
            self.save_reconstructions_grid_3D_MAP_DE()
        self.make_histograms_grid_3D_MAP_DE()
        print "-Reconstruction using 3D maximum-a-posteriori (DepthEmbedding + MLEE model)"
        if self.load_reconstructions_grid_3D_MAP_MLEEM() is None: 
            self.reconstruct_grid_3D_MAP_MLEEM()
            self.save_reconstructions_grid_3D_MAP_MLEEM()
        self.make_histograms_grid_3D_MAP_MLEEM()
        print "Computing Bias and Variance"
        self.compute_bias_and_variance_grid() 
        print "Visualizing results"
        self.visualize_results_grid() 
        print "TestCmice Done"

    def get_data_grid(self, x,y, N=10000): 
        data, depth = simulation_beam(self.forward_model_true[x,y,:,:], N=N)
        return data, depth

    def print_percentage(self,value,max_value):
        print "Progress: %d%%"%int32((100*(value+1))/(max_value))
        



if __name__ == "__main__": 
    print "--- DepthEmbedding ---"
    print "-- Simulation of calibration of cMiCE PET camera with 15mm-thick crystal, without resampling of the forward model.."
    
    cmice = CmiceSimulation_15mm(nx_resample=26, ny_resample=49, nz_resample=20, n_neighbours=12, n_training=2700, n_testing=10000, nz=24, )
    cmice.run() 
    
    print "-- Simulation of calibration of cMiCE PET camera with 15mm-thick crystal, with resampling of the forward model.."
    cmice = CmiceSimulation_15mm(nx_resample=52, ny_resample=98, nz_resample=40, n_neighbours=12, n_training=2700, n_testing=10000, nz=24, )
    cmice.run() 

    print "-- Done"


