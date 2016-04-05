
# This software is part of Occiput
# March 2016
# Stefano Pedemonte
# Martinos Center for Biomedical Imaging, MGH, Boston, MA


from sklearn import manifold 
import scipy
import scipy.io
from numpy import zeros, ones, sort, unravel_index, repeat, sum, where, squeeze
from numpy import log, tile, float32, argsort, int32, histogram, linspace, round, exp
import pylab
pylab.ion()



EPS = 1e-9


class ReconstructorMAP(): 
    """Maximum a-posteriori estimation of the coordinates of interaction and energy
    of a gamma photon (2D and 3D)."""
    def __init__(self, forward_model): 
        self.forward_model = forward_model 
        self._nd = len(self.forward_model.shape)-1
        self.prior = None

    def is_3D(self):
        return self._nd == 3 

    def set_prior(self, prior): 
        self.prior = prior 

    def reconstruct(self, data, batch_size=0): 
        N = data.shape[0]
        data_size = data.shape[1]
        if batch_size is 0: 
            batch_size = N
        if self.is_3D(): 
            # 3D reconstruction 
            Nx = self.forward_model.shape[0]
            Ny = self.forward_model.shape[1]
            Nz = self.forward_model.shape[2]
            L = self.forward_model.reshape(Nx*Ny*Nz,data_size).T
            logL = log(L+EPS) 
            logSumL = log(L.sum(0)) 
            if self.prior is not None: 
                logPrior = log(self.prior)
                # if prior on depth (not on x,y,z), then repeat for each x,y location
                if len(logPrior.shape) == 1: 
                    logPrior = tile(logPrior,(Nx,Ny,1))
            else: 
                logPrior = 0 

            n_batches = N / batch_size
            X = zeros(N) 
            Y = zeros(N) 
            Z = zeros(N)
            Energy = zeros(N)

            for i_batch in range(n_batches): 
                D = data[i_batch*batch_size:(i_batch+1)*batch_size,:]
                logD = tile(logSumL,(batch_size,1)) * tile(data.sum(1),(1024,1)).T
                loglikelihood = D.dot(logL) - logD + logPrior
                index = int32(argsort(loglikelihood,axis=1))
                I = index[:,-1]
                [x,y,z] = unravel_index(I,[Nx,Ny,Nz])
                X[i_batch*batch_size:(i_batch+1)*batch_size] = x
                Y[i_batch*batch_size:(i_batch+1)*batch_size] = y
                Z[i_batch*batch_size:(i_batch+1)*batch_size] = z 
                suml = self.forward_model[x,y,z,:].sum(1)
                Energy[i_batch*batch_size:(i_batch+1)*batch_size] = sumD/suml
            return X, Y, Z, Energy
        else: 
            # 2D reconstruction 
            Nx = self.forward_model.shape[0]
            Ny = self.forward_model.shape[1]
            L = self.forward_model.reshape(Nx*Ny,data_size).T
            logL = log(L+EPS) 
            logSumL = log(L.sum(0)+EPS) 
            if self.prior is not None: 
                logPrior = log(self.prior) 
            else: 
                logPrior = 0 

            n_batches = N / batch_size
            X = zeros(N) 
            Y = zeros(N)
            Energy = zeros(N) 

            for i_batch in range(n_batches): 
                D = data[i_batch*batch_size:(i_batch+1)*batch_size,:]
                sumD = data.sum(1)
                logD = tile(logSumL,(batch_size,1)) * tile(sumD,(1024,1)).T
                loglikelihood = D.dot(logL) - logD + logPrior
                index = int32(argsort(loglikelihood,axis=1))
                I = index[:,-1] 
                [x,y] = unravel_index(I,[Nx,Ny])
                X[i_batch*batch_size:(i_batch+1)*batch_size] = x
                Y[i_batch*batch_size:(i_batch+1)*batch_size] = y
                suml = self.forward_model[x,y,:].sum(1)
                Energy[i_batch*batch_size:(i_batch+1)*batch_size] = (sumD+EPS)/(suml+EPS)
            return X, Y, Energy



class ReconstructorCentroid():
    """Centroid algorithm for the 2D estimation of the coordinates of interaction 
    of a gamma photon.""" 
    def __init__(self, x_detectors, y_detectors, x_max, y_max, x_min=0, y_min=0, scale=1, shift=0, exponent=1.0): 
        self.exponent = exponent
        self.x_detectors = x_detectors
        self.y_detectors = y_detectors
        self.shift = shift
        self.scale = scale
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def reconstruct(self, data): 
        data_size = data.shape[1]
        N = data.shape[0]

        X_detectors = tile(self.x_detectors, (N,1))
        Y_detectors = tile(self.y_detectors, (N,1)) 

        Energy = data.sum(1)

        X = (data**self.exponent * X_detectors).sum(1) / Energy
        Y = (data**self.exponent * Y_detectors).sum(1) / Energy
        
        X = round((X + self.shift)*self.scale)
        Y = round((Y + self.shift)*self.scale)
        X[X<=self.x_min]=self.x_min
        Y[Y<=self.y_min]=self.y_min 
        X[X>=self.x_max]=self.x_max 
        Y[Y>=self.y_max]=self.y_max 

        return X,Y, Energy




def get_cumulative_prior(prior):  
    cumulative_probability = zeros(prior.shape)
    cumulative_probability[0] = prior[0]
    for i in range(1,len(prior)): 
        cumulative_probability[i] = cumulative_probability[i-1] + prior[i]
    cumulative_probability = cumulative_probability / cumulative_probability[-1]  
    return cumulative_probability 



class BeerLambert(): 
    """BeerLambert law of absorption."""
    def __init__(self, alpha_inv_cm=0.83, n_bins=16, bin_size_cm=1.0/16): 
        self.alpha = alpha_inv_cm
        self.n_bins = n_bins 
        self.bin_size = bin_size_cm 
    
    def get_probability_mass_function(self):
        iz = int32(range(self.n_bins)) 
        probability = (exp(-self.alpha*iz*self.bin_size)-exp(-self.alpha*(iz+1)*self.bin_size))/ (1.0-exp(-self.alpha*self.n_bins*self.bin_size))
        return probability

    def get_cumulative_probability(self): 
        probability = self.get_probability_mass_function() 
        return get_cumulative_prior(probability)




class EnergySpectrum():
    """Data structure to compute the histogram of the energy of the gamma photons."""
    def __init__(self, min_energy=0, max_energy=1000, n_bins=100):
        self.histogram = None 
        self.energy = None
        self.min_energy = min_energy
        self.max_energy = max_energy 
        self.n_bins = n_bins 
    
    def add_data(self, energies): 
        [hist, energy] = histogram(energies, bins=self.n_bins, range=(self.min_energy,self.max_energy))
        if self.histogram is None: 
            self.histogram = hist
        else: 
            self.histogram = self.histogram + hist
        self.energy = energy 
    
    def get_spectrum(self): 
        return self.histogram, self.energy

    def show(self): 
        pylab.figure()
        pylab.plot(self.histogram, self.energy)




class HistogramCoordinates(): 
    """Histogram of the coordinates of interaction in 2D and 3D."""
    def __init__(self, nx, ny, nz=0): 
        self.nx = nx
        self.ny = ny
        self.nz = nz 
        if self.nz == 0:
            self.dimensions = 2
            self.histogram = zeros([nx,ny]) 
        else: 
            self.dimensions = 3
            self.histogram = zeros([nx,ny,nz]) 

    def is_2D(self):
        return self.dimensions == 2

    def add_data(self, coordinates): 
        if self.is_2D(): 
            for i in range(len(coordinates[0])): 
                self.histogram[coordinates[0][i],coordinates[1][i]] += 1
        else: 
            for i in range(len(coordinates[0])): 
                self.histogram[coordinates[0][i],coordinates[1][i],coordinates[2][i]] += 1        

    def get_histogram(self): 
        return self.histogram 

    def show(self, depth=None, interpolation='nearest'):
        """Display histogram using pylab. If histogram is 3D, 'depth' indicates depth. If 
        'depth = None', the function displays the mean of the histograms over the depth. """
        if self.is_2D(): 
            image = self.get_histogram() 
        else: 
            if depth is None: 
                image = self.get_histogram().mean(2)
        pylab.figure()
        pylab.imshow(image, interpolation=interpolation)
        pylab.show()



class LikelihoodFilter(): 
    """Filter events based on the likelihood value. """
    def __init__(self, forward_model_2D):
        self.forward_model_2D = forward_model_2D

    def filter(self, data, x, y, method="match_coordinates"): 
        # MLE 2D
        reconstructor = ReconstructorMAP(self.forward_model_2D)
        [x_mle_2D, y_mle_2D, energy_mle_2D] = reconstructor.reconstruct(data) 
        # Filter 
        if method=="match_coordinates": 
            indexes = where((x_mle_2D==x) & (y_mle_2D==y))
            data_filtered = data[indexes,:]
        return squeeze(data_filtered)



class Model2D(): 
    """Simple estimation of the forward model parameters of a gamma camera by averaging of the 
    measurement vectors."""
    def __init__(self): 
        self.n_detectors = 0 
        self.calibration_data = None 
        self.n_training = 0
        self.n_detectors = 0 

    def set_calibration_data(self, data):
        if not self._check_data(data): 
            print "Model2D: Calibration data does not have the correct size. "
            return 
        self.calibration_data = data 
        self.n_training  = data.shape[0]
        self.n_detectors = data.shape[1]

    def estimate_forward_model(self): 
        forward_model = zeros([1,self.n_detectors])
        for i in range(self.n_training): 
            forward_model += self.calibration_data[i,:] 
        forward_model /= self.n_training 
        # Don't allow negative values 
        forward_model[forward_model<EPS]=EPS 
        self.forward_model = forward_model
        return self.forward_model

    def _check_data(self, data): 
        return len(data.shape)==2
        
    def visualize_model(self, x, y, reshape=(8,8)): 
        m = self.forward_model[x,y,:].reshape(reshape)
        pylab.imshow(m)



class ModelDepthEmbedding(): 
    """DepthEmbedding algorithm for the estimation of the characteristics of a gamma camera."""
    def __init__(self, nz=16, n_neighbours=12, n_components=1, lle_method='standard'): 
        self.nz = nz
        self.n_neighbours = n_neighbours
        self.n_components = n_components
        self.lle_method = lle_method
        self.calibration_data = None 
        self.n_training = 0
        self.n_detectors = 0 
        self.set_default_depth_prior() 
        
    def set_default_depth_prior(self): 
        prior = BeerLambert() 
        self.set_depth_prior(prior.get_probability_mass_function() )
        
    def set_depth_prior(self, prior): 
        if len(prior) != self.nz: 
            print "Length of depth prior vector must be equal to self.nz (%d)"%self.nz
            return 
        self.prior = prior 

    def set_calibration_data(self, data):
        if not self._check_data(data): 
            print "ModelDepthEmbedding: Calibration data does not have the correct size. "
            return 
        self.calibration_data = data 
        self.n_training  = data.shape[0]
        self.n_detectors = data.shape[1]
        
    def estimate_forward_model(self): 
        data = self.calibration_data
        N = data.shape[0]
        forward_model = zeros([self.nz,self.n_detectors]) 
        cumulative_prior = get_cumulative_prior(self.prior)

        print "1-Projecting calibration data onto the manifold.."
        self.manifold = manifold.LocallyLinearEmbedding(self.n_neighbours, n_components=self.n_components, method=self.lle_method) 
        data_1d = self.manifold.fit_transform(data) 

        print "2-Sorting the manifold.."
        data_1d = data_1d-data_1d.min()
        data_1d = data_1d/data_1d.max()
        data_1d = data_1d-data_1d.mean()
        data_1d_s = sort(data_1d)
        s = argsort(data_1d) 

        print "3-Determining the orientation of the manifold.."
#        manifold_density = 1.0 / conv(sqrt(mean_distance_points[s]),ones(1,10*neighbours), 'valid') 
        manifold_density = ones(10);  #FIXME 

        if (manifold_density[0] < manifold_density[-1]): 
            invert = 1 
            data_1d = -data_1d
            data_1d_s = -flipud(data_1d_s)
            s = flipud(s) 
        else: 
            invert = 0 

        print "4-Depth mapping.."   
        # Depth mapping - determine boundaries 
        boundaries = zeros(self.nz+1) 
        boundaries[0] = data_1d_s[0]
        indexes = int32(round(cumulative_prior*(N-1)))
        boundaries[1::] = squeeze(data_1d_s[indexes])

        # Depth mapping - determine membership
        membership = zeros(N)
        for i in range(N):
            for z in range(self.nz):
                if ( data_1d[i]>=boundaries[z] ) and ( data_1d[i]<=boundaries[z+1] ): 
                    membership[i]= z

        self.membership = membership
        self.data_1d = data_1d
        self.data_1d_s = data_1d_s
        self.s = s 
        self.invert = invert
        self.manifold_density = manifold_density
        
        # Don't allow negative values 
        forward_model[forward_model<EPS]=EPS 
        self.forward_model = forward_model
        return forward_model 

    def visualize_manifold(self, nd=3): 
        data = self.calibration_data
        man = manifold.LocallyLinearEmbedding(self.n_neighbours, n_components=nd, method=self.lle_method) 
        data_nd = man.fit_transform(data) 

        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data_nd[:,0], data_nd[:,1], data_nd[:,2])
        ax.set_xlabel('r1')
        ax.set_ylabel('r2')
        ax.set_zlabel('r3')
        plt.show()

    def visualize_model(self, x, y, depth=None, reshape=(8,8)): 
        if depth is None: 
            m = self.forward_model[x,y,:,:].mean(0).reshape(reshape)
        else: 
            m = self.forward_model[x,y,depth,:].reshape(reshape)
        pylab.imshow(m)

    def _check_data(self, data): 
        return len(data.shape)==2


    

class ModelMLEEM(): 
    """Algorithm for the estimation of the characteristics of a gamma camera based on 
    Expectation Maximization Maximum-likelihood-estimation."""
    def __init__(self, initial_model):
        self.caibration_data = None 
        self.prior = None
        self.set_initial_model(initial_model)

    def estimate_forward_model(self, method="hard", n_max_iterations=10, toll=1e-3): 
        nz = self.initial_model.shape[0]
        n_detectors = self.initial_model.shape[1]
        forward_model = self.initial_model 
        if method is "hard": 
            for i in range(n_max_iterations): 
                pass # FIXME
        elif method is "soft": 
            for i in range(n_max_iterations): 
                pass # FIXME
        self.forward_model = forward_model 
        return forward_model 

    def set_calibration_data(self, calibration_data): 
        self.calibration_data = calibration_data 

    def set_initial_model(self, initial_model): 
        if self.prior != None: 
            if initial_model.shape[0] != len(self.prior): 
                 print "The number of depth bins in the forward model must match the length of the multinomial prior (%d)."%len(self.prior)
                 return
        self.initial_model = initial_model 

    def set_depth_prior(self, prior): 
        if len(prior) != self.initial_model.shape[0]: 
            print "Length of depth prior vector must be equal to the numeber of depth bins in the forward model (%d)."%self.self.initial_model.shape[0]
            return 
        self.prior = prior 

    def visualize_model(self, x, y, depth=None, reshape=(8,8)): 
        if depth is None: 
            m = self.forward_model[x,y,:,:].mean(0).reshape(reshape)
        else: 
            m = self.forward_model[x,y,depth,:].reshape(reshape)
        pylab.imshow(m)





def get_data_cmice(x,y,path="./data_ftp/cmice_data/20140508_ZA0082/test_data/"): 
    """cMiCe camera data: load from file the calibration or test data for a given beam position. 
    Return a ndarray."""
    filename = path + "/ROW%s_COL%s.mat"%(str(x+1).zfill(2), str(y+1).zfill(2))
    data = scipy.io.loadmat(filename)['data']
    data[data<0] = 0
    return float32(data.T)




class TestCmice(): 
    """Estimation of the forward model of the cMiCe PET camera and reconstruction using various 2D and 3D methods. """
    def __init__(self, nx=32, ny=32, nz=16, n_neighbours=12): 
        self.enable_cache() 
        self.nx = nx 
        self.ny = ny 
        self.nz = nz 
        self.n_neighbours = n_neighbours
        self.n_detectors = 64 
        self.scintillator_attenuation_coefficient = 0.83 #[cm-1]
        self.scintillator_thickness = 0.8 #[cm]
        self._set_depth_prior()
        self.forward_model_2D = None 
        self.forward_model_3D_DE = None 
        self.forward_model_3D_MLEEM = None 
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
        prior_model = BeerLambert(self.scintillator_attenuation_coefficient, self.nz, self.scintillator_thickness/self.nz)
        self.prior = prior_model.get_probability_mass_function()

    def load_forward_model_2D(self, filename='./model_cmice_2D.mat'): 
        try: 
            model = scipy.io.loadmat(filename)['model_cmice_2D']
        except: 
            return None
        self.forward_model_2D = model
        return model

    def load_forward_model_DE(self, filename='./model_cmice_DE.mat'): 
        try: 
            model = scipy.io.loadmat(filename)['model_cmice_DE']
        except: 
            return None
        self.forward_model_3D_DE = model
        return model

    def load_forward_model_MLEEM(self, filename='./model_cmice_MLEEM.mat'): 
        try: 
            model = scipy.io.loadmat(filename)['model_cmice_MLEEM']
        except: 
            return None
        self.forward_model_3D_MLEEM = model
        return model

    def save_forward_model_2D(self, filename='./model_cmice_2D.mat'): 
        scipy.io.savemat(filename, {'model_cmice_2D':self.forward_model_2D})

    def save_forward_model_DE(self, filename='./model_cmice_DE.mat'): 
        scipy.io.savemat(filename, {'model_cmice_DE':self.forward_model_DE})
 
    def save_forward_model_MLEEM(self, filename='./model_cmice_MLEEM.mat'): 
        scipy.io.savemat(filename, {'model_cmice_MLEEM':self.forward_model_MLEEM})

    def estimate_forward_model_2D(self): 
        model = zeros([self.nx, self.ny, self.n_detectors])
        for ix in range(self.nx): 
#        for ix in [1]:
            for iy in range(self.ny): 
                calibration_data = get_data_cmice(ix,iy)
                model_estimator = Model2D()
                model_estimator.set_calibration_data(calibration_data)
                model[ix,iy,:] = model_estimator.estimate_forward_model()
        self.forward_model_2D = model 
        return self.forward_model_2D

    def estimate_forward_model_DE(self): 
        model = zeros([self.nx, self.ny, self.nz, self.n_detectors])
#        for ix in range(self.nx): 
        for ix in [1]:
            for iy in range(self.ny): 
                calibration_data = get_data_cmice(ix,iy)
                filter = LikelihoodFilter(self.forward_model_2D)
                calibration_data_filtered = filter.filter(calibration_data, ix,iy) 
                model_estimator = ModelDepthEmbedding(nz=self.nz, n_neighbours=self.n_neighbours)
                model_estimator.set_calibration_data(calibration_data_filtered)
                model_estimator.set_depth_prior(self.prior)
                model[ix,iy,:,:] = model_estimator.estimate_forward_model() 
        self.forward_model_3D_DE = model 
        return self.forward_model_3D_DE
    
    def estimate_forward_model_MLEEM(self): 
        model = zeros([self.nx, self.ny, self.nz, self.n_detectors])
#        for ix in range(self.nx): 
        for ix in [1]:
            for iy in range(self.ny): 
                calibration_data = get_data_cmice(ix,iy)
                filter = LikelihoodFilter(self.forward_model_2D)
                calibration_data_filtered = filter.filter(calibration_data, ix,iy) 
                model_estimator = ModelMLEEM(initial_model=self.forward_model_3D_DE[ix,iy,:,:])
                model_estimator.set_calibration_data(calibration_data_filtered) 
                model_estimator.set_depth_prior(self.prior) 
                model[ix,iy,:,:] = model_estimator.estimate_forward_model() 
        self.forward_model_3D_MLEEM = model 
        return self.forward_model_3D_MLEEM

    def load_test_grid(self, x_locations, y_locations): 
        self.grid = []
        for ix in x_locations: 
 #       for ix in [1]: 
            grid_row = [] 
            for iy in y_locations: 
                data = get_data_cmice(ix,iy)
                grid_row.append(data)
            self.grid.append(grid_row) 
        self.grid_shape = (len(x_locations),len(y_locations))
        return self.grid

    def reconstruct_grid_centroid(self): 
        self.coordinates_centroid = [] 
        x_detectors = tile(linspace(0,9,8),(1,8))[0] - 4.5 
        y_detectors = repeat(linspace(0,9,8),8,axis=0) - 4.3
        reconstructor = ReconstructorCentroid(x_detectors=x_detectors, y_detectors=y_detectors, x_max=self.nx-1, y_max=self.ny-1, shift=2.7, scale=5.3, exponent=1.0)  
        self.histogram_centroid = HistogramCoordinates(self.nx, self.ny)
        self.spectrum_centroid = EnergySpectrum() 
        for ix in range(self.grid_shape[0]): 
            row = [] 
            for iy in range(self.grid_shape[1]):
                data = self.grid[ix][iy]
                [xrec,yrec,energyrec] = reconstructor.reconstruct(data) 
                row.append([xrec,yrec])
                self.histogram_centroid.add_data([xrec,yrec])
                self.spectrum_centroid.add_data(energyrec)
            self.coordinates_centroid.append(row)
        return self.coordinates_centroid

    def reconstruct_grid_2D_MAP(self): 
        self.coordinates_2D_MAP = []
        reconstructor = ReconstructorMAP(self.forward_model_2D)  
        self.histogram_2D_MAP = HistogramCoordinates(self.nx, self.ny)
        self.spectrum_2D_MAP = EnergySpectrum()
        for ix in range(self.grid_shape[0]): 
            row = []
            for iy in range(self.grid_shape[1]):
                data = self.grid[iy][ix]
                [xrec,yrec,energyrec] = reconstructor.reconstruct(data) 
                row.append([xrec,yrec])
                self.histogram_2D_MAP.add_data([xrec,yrec])
                self.spectrum_2D_MAP.add_data(energyrec)
            self.coordinates_2D_MAP.append(row)
        return self.coordinates_2D_MAP

    def reconstruct_grid_3D_MAP_DE(self): 
        self.coordinates_3D_MAP_DE = []
        reconstructor = ReconstructorMAP(self.forward_model_3D_DE)  
        reconstructor.set_prior(self.prior)
        self.histogram_3D_MAP_DE = HistogramCoordinates(self.nx, self.ny, self.nz)
        self.spectrum_3D_MAP_DE = EnergySpectrum()
#        for ix in range(self.grid_shape[0]): 
        for ix in [1]:
            row = []
            for iy in range(self.grid_shape[1]):
                data = self.grid[iy][ix]
                [xrec,yrec,zrec,energyrec] = reconstructor.reconstruct(data) 
                row.append([xrec,yrec,zrec])
                self.histogram_3D_MAP_DE.add_data([xrec,yrec,zrec])
                self.spectrum_3D_MAP_DE.add_data(energyrec)
            self.coordinates_3D_MAP_DE.append(row)
        return self.coordinates_3D_MAP_DE

    def reconstruct_grid_3D_MAP_MLEEM(self): 
        self.coordinates_3D_MAP_MLEEM = []
        reconstructor = ReconstructorMAP(self.forward_model_3D_MLEEM)  
        reconstructor.set_prior(self.prior)
        self.histogram_3D_MAP_MLEEM = HistogramCoordinates(self.nx, self.ny, self.nz)
        self.spectrum_3D_MAP_MLEEM = EnergySpectrum()
#        for ix in range(self.grid_shape[0]): 
        for ix in [1]:
            row = []
            for iy in range(self.grid_shape[1]):
                data = self.grid[iy][ix]
                [xrec,yrec,zrec,energyrec] = reconstructor.reconstruct(data) 
                row.append([xrec,yrec,zrec])
                self.histogram_3D_MAP_MLEEM.add_data([xrec,yrec,zrec])
                self.spectrum_3D_MAP_MLEEM.add_data(energyrec)
            self.coordinates_3D_MAP_MLEEM.append(row)
        return self.coordinates_3D_MAP_MLEEM

    def compute_bias_and_variance(self): 
        if self.coordinates_centroid is not None: 
            pass 
        if self.coordinates_2D_MAP is not None: 
            pass 
        if self.coordinates_3D_MAP_DE is not None: 
            pass 
        if self.coordinates_3D_MAP_MLEEM is not None: 
            pass 

    def visualize_results(self): 
        if self.coordinates_centroid is not None: 
            self.histogram_centroid.show()
        if self.coordinates_2D_MAP is not None: 
            self.histogram_2D_MAP.show()
        if self.coordinates_3D_MAP_DE is not None: 
            self.histogram_3D_MAP_DE.show()
        if self.coordinates_3D_MAP_MLEEM is not None: 
            self.histogram_3D_MAP_MLEEM.show()

    def run(self): 
        print "-Estimating the forward model 2D .."
        if self.load_forward_model_2D() is None: 
            self.estimate_forward_model_2D() 
        print "-Estimating the forward model DepthEmbedding .."
#        if self.load_forward_model_DE() is None: 
#            self.estimate_forward_model_DE() 
        print "-Estimating the forward model MLEEM .."
#        if self.load_forward_model_MLEEM() is None:
#            self.estimate_forward_model_MLEEM() 
        print "-Loading 2D test grid"
        self.load_test_grid([0,5,10,15,20,25,30], [0,5,10,15,20,25,30])
        print "-Reconstruction using centroid algorithm"
        self.reconstruct_grid_centroid()
        print "-Reconstruction using 2D maximum-a-posteriori"
        #self.reconstruct_grid_2D_MAP()
        print "-Reconstruction using 3D maximum-a-posteriori (DepthEmbedding model)"
        #self.reconstruct_grid_3D_MAP_DE()
        print "-Reconstruction using 3D maximum-a-posteriori (DepthEmbedding + MLEE model)"
        #self.reconstruct_grid_3D_MAP_MLEEM()

        print "Computing Bias and Variance"
        #self.compute_bias_and_variance() 
        print "Visualizing results"
        self.visualize_results() 
        print "TestCmice Done"

    def enable_cache(self): 
        self._cache = True
    
    def disable_cache(self): 
        self._cache = False 




def get_data_simulation(filename = "./sysmat.mat"): 
    """Load from file the ground-truth forward model of a simulated monolithic gamma camera. """
    data = scipy.io.loadmat(filename)
    data[data<0] = 0
    return data['sysmat'].T



class TestSimulation(TestCmice): 
    """Estimation of the forward model of a simulated monolithic PET camera 
    and reconstruction using various 2D and 3D methods. """
    def run(self): 
        pass 
    




if __name__ == "__main__": 
    print "--- DepthEmbedding ---"
    print "-- Running cMiCe PET camera calibration .."
    cmice = TestCmice()
    cmice.run() 
    
    print "-- Running monolithic gamma camera simulation .."
    simulation = TestSimulation()
    sumulation.run()
    
    print "-- Done"


