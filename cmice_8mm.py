
from depth_embedding import ReconstructorMAP, ReconstructorCentroid, get_cumulative_prior
from depth_embedding import BeerLambert, EnergySpectrum, HistogramCoordinates, LikelihoodFilter
from depth_embedding import ModelInterpolator, Model2D, ModelDepthEmbedding, ModelMLEEM, StatsCoordinates, model_error

import scipy
import scipy.io
from numpy import zeros, ones, sort, unravel_index, repeat, sum, where, squeeze, fliplr, flipud
from numpy import log, tile, float32, argsort, int32, histogram, linspace, round, exp, convolve, sqrt, mgrid
import numpy as np
import copy
import pylab as pl 


BATCH_SIZE = 128  # Number of events reconstructed in parallel (i.e. in a single batch). 
                  # If set to 0, all events are reconstructed at once: faster but may starve memory when 
                  # using large reconstruction grids (e.g. when up-sampling the forward model). 

def get_data_cmice_grid(x,y,path="./"): 
    """cMiCe: data acquired on a regular grid x,y."""
    path = path + "/20140508_ZA0082/test_data/"
    filename = path + "/ROW%s_COL%s.mat"%(str(x+1).zfill(2), str(y+1).zfill(2))
    data = scipy.io.loadmat(filename)['data']
    data[data<0] = 0
    return float32(data.T)




class Cmice_8mm(): 
    """Estimation of the forward model of the cMiCe PET camera and reconstruction using various 2D and 3D methods. """
    def __init__(self, n_neighbours=12, n_training=5000, nz=16, nx_resample=32, ny_resample=32, nz_resample=16): 
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
        prior_model = BeerLambert(self.scintillator_attenuation_coefficient, self.nz, self.scintillator_thickness/self.nz)
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



    def estimate_forward_model_2D(self): 
        model = zeros([self.nx, self.ny, self.n_detectors])
        for ix in range(self.nx): 
            self.print_percentage(ix,self.nx)
            for iy in range(self.ny): 
                calibration_data = self.get_data_grid(ix,iy)
                model_estimator = Model2D()
                model_estimator.set_calibration_data(calibration_data)
                model[ix,iy,:] = model_estimator.estimate_forward_model()
        self.forward_model_2D = model 
        return self.forward_model_2D

    def estimate_forward_model_DE(self): 
        model = zeros([self.nx, self.ny, self.nz, self.n_detectors])
        for ix in range(self.nx): 
            self.print_percentage(ix,self.nx)
            for iy in range(self.ny): 
                calibration_data = self.get_data_grid(ix,iy)
                filter = LikelihoodFilter(self.forward_model_2D)
                calibration_data_filtered, rejected = filter.filter(calibration_data, ix,iy, points = self.n_training) 
                model_estimator = ModelDepthEmbedding(nz=self.nz, n_neighbours=self.n_neighbours)
                model_estimator.set_calibration_data(calibration_data_filtered)
                model_estimator.set_depth_prior(self.prior)
                model[ix,iy,:,:] = model_estimator.estimate_forward_model() 
        self.forward_model_DE = model 
        return self.forward_model_DE
    
    def estimate_forward_model_MLEEM(self): 
        model = zeros([self.nx, self.ny, self.nz, self.n_detectors])
        for ix in range(self.nx): 
            self.print_percentage(ix,self.nx)
            for iy in range(self.ny): 
                calibration_data = self.get_data_grid(ix,iy)
                filter = LikelihoodFilter(self.forward_model_2D)
                calibration_data_filtered, rejected = filter.filter(calibration_data, ix,iy, points = self.n_training) 
                model_estimator = ModelMLEEM(initial_model=self.forward_model_DE[ix,iy,:,:])
                model_estimator.set_calibration_data(calibration_data_filtered) 
                model_estimator.set_depth_prior(self.prior) 
                model[ix,iy,:,:] = model_estimator.estimate_forward_model() 
        self.forward_model_MLEEM = model 
        return self.forward_model_MLEEM


    def resample_forward_model_2D(self, grid_size, interpolation='cubic', noise_sigma=0.5): 
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

    def resample_forward_model_DE(self, grid_size, interpolation='cubic', noise_sigma=0.5): 
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

    def resample_forward_model_MLEEM(self, grid_size, interpolation='cubic', noise_sigma=0.5): 
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
        self.grid_locations = [tile(x_locations,(len(y_locations),1)).transpose(), tile(int32(y_locations).reshape((len(y_locations),1)),(1,len(x_locations))).transpose()]
        return self.grid


            
    def reconstruct_grid_centroid(self, shiftx=2.9, shifty=2.5, scale=8.0): 
        self.coordinates_centroid = [] 
        self.energy_centroid = []
        x_detectors = tile(linspace(0,9,8),(1,8))[0] - 4.5 
        y_detectors = repeat(linspace(0,9,8),8,axis=0) - 4.3
        reconstructor = ReconstructorCentroid(x_detectors=x_detectors, y_detectors=y_detectors, x_max=self.nx_resample-1, y_max=self.ny_resample-1, shiftx=shiftx, shifty=shifty, scale=scale, exponent=1.0)  
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
                energyrec = self.energy_centroid[ix][iy]
                self.histogram_centroid.add_data([xrec,yrec])
                self.spectrum_centroid.add_data(energyrec)

    def reconstruct_grid_2D_MAP(self): 
        self.coordinates_2D_MAP = []
        self.energy_2D_MAP = []
        self.likelihood_2D_MAP = []
        reconstructor = ReconstructorMAP(self.forward_model_2D)  
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
                #print ix,iy,energyrec
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


    def filter_events_2D_MAP(self, fraction=0.5): 
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
            stats_centroid = StatsCoordinates(self.coordinates_centroid, grid_locations, [(0,self.nx_resample),(0,self.ny_resample)], "Centroid", scale=(50.0/self.nx_resample, 50.0/self.ny_resample, 8.0/self.nz_resample))
            stats_centroid.print_summary()
        if self.coordinates_2D_MAP_filtered is not None: 
            stats_2D_MAP = StatsCoordinates(self.coordinates_2D_MAP_filtered, grid_locations, [(0,self.nx_resample),(0,self.ny_resample)], "2D MAP", scale=(50.0/self.nx_resample, 50.0/self.ny_resample, 8.0/self.nz_resample))
            stats_2D_MAP.print_summary()
        if self.coordinates_3D_MAP_DE_filtered is not None: 
            stats_3D_MAP_DE = StatsCoordinates(self.coordinates_3D_MAP_DE_filtered, grid_locations, [(0,self.nx_resample),(0,self.ny_resample),(2,self.nz_resample-2)], "3D MAP DepthEmbedding", scale=(50.0/self.nx_resample, 50.0/self.ny_resample, 8.0/self.nz_resample))
            stats_3D_MAP_DE.print_summary()
            #stats_3D_MAP_DE.plot_depth_density(6)
        if self.coordinates_3D_MAP_MLEEM_filtered is not None: 
            stats_3D_MAP_MLEEM = StatsCoordinates(self.coordinates_3D_MAP_MLEEM_filtered, grid_locations, [(0,self.nx_resample),(0,self.ny_resample),(2,self.nz_resample-2)], "3D MAP DepthEmbedding+MLEEM", scale=(50.0/self.nx_resample, 50.0/self.ny_resample, 8.0/self.nz_resample))
            stats_3D_MAP_MLEEM.print_summary()
            #stats_3D_MAP_MLEEM.plot_depth_density(6)
            

    def visualize_results_grid(self, n_points=4000, vmax=1000): 
        # Visualize manifold: 
        if n_points is None or n_points is 0:
            n_points = self.n_training  
        ix = 15
        iy = 15
        Ix = ix*1.0 / self.nx * self.nx_resample
        Iy = iy*1.0 / self.ny * self.ny_resample
        calibration_data = self.get_data_grid(ix,iy) 
        filter = LikelihoodFilter(self.forward_model_2D)
        calibration_data_filtered, rejected = filter.filter(calibration_data, Ix,Iy, points=n_points) 
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
        #self.load_test_grid([1,6,11,16,21,26,31], [1,6,11,16,21,26,31])
        self.load_test_grid([2,9,16,23,30], [2,9,16,23,30])
        print "-Reconstruction using centroid algorithm"
        self.reconstruct_grid_centroid(shiftx=2.8, shifty=2.8, scale=5.2*self.nx_resample / 32.0)
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
        print "Visualizing results"
        self.visualize_results_grid() 
        print "TestCmice Done"

    def get_data_grid(self, x,y):
        return get_data_cmice_grid(x,y, path=self.input_data_path)

    def print_percentage(self,value,max_value):
        print "Progress: %d%%"%int32(100*value/(max_value-1))






if __name__ == "__main__": 
    print "--- DepthEmbedding ---"
    print "-- Calibration of cMiCE PET camera with 8mm-thick crystal, without resampling of the forward model.."
    cmice = Cmice_8mm(nx_resample=32, ny_resample=32, nz_resample=16, n_neighbours=12, n_training=2500, nz=16)
    cmice.run() 
    
    print "-- Calibration of cMiCE PET camera with 8mm-thick crystal, with resampling of the forward model.."
    cmice = Cmice_8mm(nx_resample=128, ny_resample=128, nz_resample=32, n_neighbours=12, n_training=2500, nz=32)
    cmice.run() 

    print "-- Done"

