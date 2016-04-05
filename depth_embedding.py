
# This software is part of Occiput
# March 2016
# Stefano Pedemonte
# Martinos Center for Biomedical Imaging, MGH, Boston, MA


from sklearn import manifold 
import scipy
import scipy.io
from numpy import zeros, ones, sort, unravel_index, log, tile, float32, argsort
import pylab
pylab.ion()



EPS = 1e-9


class ReconstructorMAP(): 
    """Maximum a-posteriori estimation of the coordinates of interaction and energy
    of a gamma photon (2D and 3D)."""
    def __init__(self, forward_model): 
        self.forward_model = forward_model 
        self._nd = len(self.forward_model.shape)-1

    def is_3D(self):
        return self._nd == 3 

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
            logL = log(L+EPS);
            sumL = tile(L.sum(0),(batch_size,1)); 

            n_batches = N / batch_size; 
            X = zeros(N); 
            Y = zeros(N); 
            Z = zeros(N); 

            for i_batch in range(n_batches): 
                like = data[i_batch*batch_size:(i_batch+1)*batch_size,:].dot(logL) - sumL; 
                index = argsort(like,axis=1) 
                I = index[:,1]; 
                [x,y,z] = unravel_index(I,[Nx,Ny,Nz]); 
                X[i_batch*batch_size:(i_batch+1)*batch_size] = x;
                Y[i_batch*batch_size:(i_batch+1)*batch_size] = y; 
                Z[i_batch*batch_size:(i_batch+1)*batch_size] = z; 
            return X,Y,Z
        else: 
            # 2D reconstruction 
            Nx = self.forward_model.shape[0]
            Ny = self.forward_model.shape[1]
            L = self.forward_model.reshape(Nx*Ny,data_size).T
            logL = log(L+EPS);
            sumL = tile(L.sum(0),(batch_size,1)); 

            n_batches = N / batch_size; 
            X = zeros(N); 
            Y = zeros(N); 

            for i_batch in range(n_batches): 
                like = data[i_batch*batch_size:(i_batch+1)*batch_size,:].dot(logL) - sumL; 
                index = argsort(like,axis=1) 
                I = index[:,1]; 
                [x,y] = unravel_index(I,[Nx,Ny]); 
                X[i_batch*batch_size:(i_batch+1)*batch_size] = x;
                Y[i_batch*batch_size:(i_batch+1)*batch_size] = y; 
            return X,Y



class ReconstructorCentroid():
    """Centroid algorithm for the 2D estimation of the coordinates of interaction 
    of a gamma photon.""" 
    def __init__(self, width=1.0, height=1.0, exponent=1.0): 
        self.exponent = 1.0
        self.width = width
        self.height = height 

    def reconstruct(self, data): 
        data_size = data.shape[0]
        N = data.shape[1]
        X = zeros(N)
        Y = zeros(N)
        return X,Y




class BeerLambert(): 
    """BeerLambert law of absorption."""
    def __init__(self, alpha_inv_cm=0.83, n_bins=16, bin_size_cm=1.0/16): 
        self.alpha_inv_cm = alpha_inv_cm
        self.n_bins = n_bins 
        self.bin_size_cm = bin_size_cm 
    
    def get_probability_mass_function(self): 
        probability = ones(self.n_bins)  #FIXME
        return probability

    def get_cumulative_probability(self): 
        cumulative_probability = 0  #FIXME
        return cumulative_probability 



class Histogram(): 
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

    def show(self, depth=None):
        """Display histogram using pylab. If histogram is 3D, 'depth' indicates depth. If 
        'depth = None', the function displays the mean of the histograms over the depth. """
        if self.is_2D(): 
            image = self.get_histogram() 
        else: 
            if depth is None: 
                image = self.get_histogram().mean(2)
        pylab.figure(); 
        pylab.imshow(image)
        pylab.show()


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
            print "Calibration data does not have the correct size. "
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

    def _check_data(self, data): 
        return len(data.shape)==2



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
            print "Calibration data does not have the correct size. "
            return 
        self.calibration_data = data 
        self.n_training  = data.shape[0]
        self.n_detectors = data.shape[1]

    def estimate_forward_model(self): 
        forward_model = zeros([self.nz,self.n_detectors]) 
        return forward_model 
        
        
        print "1-Projecting calibration data onto the manifold.."
        self.manifold = manifold.LocallyLinearEmbedding(self.n_neighbours, n_components=self.n_components, method=self.lle_method) 
        self.manifold.fit_transform(self.calibration_data) 
        print "2-Depth mapping.."
        pass #FIXME
        # Don't allow negative values 
        forward_model[forward_model<EPS]=EPS 
        self.forward_model = forward_model
        return forward_model 

    def visualize_manifold(self): 
        pass #FIXME 
    
    def visualize_model(self, depth=None): 
        pass #FIXME 

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

    def visualize_model(self, depth=None):
        pass #FIXME 





def get_data_cmice(x,y,path="./data_ftp/cmice_data/20140508_ZA0082/test_data/"): 
    """cMiCe camera data: load from file the calibration or test data for a given beam position. 
    Return a ndarray."""
    filename = path + "/ROW%s_COL%s.mat"%(str(x+1).zfill(2), str(y+1).zfill(2))
    data = scipy.io.loadmat(filename)
    data[data<0] = 0
    return float32(data['data'].T)




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

    def _set_depth_prior(self): 
        prior_model = BeerLambert(self.scintillator_attenuation_coefficient, self.nz, self.scintillator_thickness/self.nz)
        self.depth_prior = prior_model.get_probability_mass_function()

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
                model_estimator = ModelDepthEmbedding(nz=self.nz, n_neighbours=self.n_neighbours)
                model_estimator.set_calibration_data(calibration_data)
                model_estimator.set_depth_prior(self.depth_prior)
                model[ix,iy,:,:] = model_estimator.estimate_forward_model() 
        self.forward_model_3D_DE = model 
        return self.forward_model_3D_DE
    
    def estimate_forward_model_MLEEM(self): 
        model = zeros([self.nx, self.ny, self.nz, self.n_detectors])
#        for ix in range(self.nx): 
        for ix in [1]:
            for iy in range(self.ny): 
                calibration_data = get_data_cmice(ix,iy)
                model_estimator = ModelMLEEM(initial_model=self.forward_model_3D_DE[ix,iy,:,:])
                model_estimator.set_calibration_data(calibration_data) 
                model_estimator.set_depth_prior(self.depth_prior) 
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
        reconstructor = ReconstructorCentroid(width=50.0, height=50.0, exponent=1.0)  
        self.histogram_centroid = Histogram(self.nx, self.ny)
        for ix in range(self.grid_shape[0]): 
            row = [] 
            for iy in range(self.grid_shape[1]):
                data = self.grid[iy][ix]
                [xrec,yrec] = reconstructor.reconstruct(data) 
                row.append([xrec,yrec])
                self.histogram_centroid.add_data([xrec,yrec])
            self.coordinates_centroid.append(row)
        return self.coordinates_centroid

    def reconstruct_grid_2D_MAP(self): 
        self.coordinates_2D_MAP = []
        reconstructor = ReconstructorMAP(self.forward_model_2D)  
        self.histogram_2D_MAP = Histogram(self.nx, self.ny)
        for ix in range(self.grid_shape[0]): 
            row = []
            for iy in range(self.grid_shape[1]):
                data = self.grid[iy][ix]
                [xrec,yrec] = reconstructor.reconstruct(data) 
                row.append([xrec,yrec])
                self.histogram_2D_MAP.add_data([xrec,yrec])
            self.coordinates_2D_MAP.append(row)
        return self.coordinates_2D_MAP

    def reconstruct_grid_3D_MAP_DE(self): 
        self.coordinates_3D_MAP_DE = []
        reconstructor = ReconstructorMAP(self.forward_model_3D_DE)  
        self.histogram_3D_MAP_DE = Histogram(self.nx, self.ny, self.nz)
        for ix in range(self.grid_shape[0]): 
            row = []
            for iy in range(self.grid_shape[1]):
                data = self.grid[iy][ix]
                [xrec,yrec,zrec] = reconstructor.reconstruct(data) 
                row.append([xrec,yrec,zrec])
                self.histogram_3D_MAP_DE.add_data([xrec,yrec,zrec])
            self.coordinates_3D_MAP_DE.append(row)
        return self.coordinates_3D_MAP_DE

    def reconstruct_grid_3D_MAP_MLEEM(self): 
        self.coordinates_3D_MAP_MLEEM = []
        reconstructor = ReconstructorMAP(self.forward_model_3D_MLEEM)  
        self.histogram_3D_MAP_MLEEM = Histogram(self.nx, self.ny, self.nz)
        for ix in range(self.grid_shape[0]): 
            row = []
            for iy in range(self.grid_shape[1]):
                data = self.grid[iy][ix]
                [xrec,yrec,zrec] = reconstructor.reconstruct(data) 
                row.append([xrec,yrec,zrec])
                self.histogram_3D_MAP_MLEEM.add_data([xrec,yrec,zrec])
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
        self.estimate_forward_model_2D() 
        print "-Estimating the forward model DepthEmbedding .."
        self.estimate_forward_model_DE() 
        print "-Estimating the forward model MLEEM .."
        self.estimate_forward_model_MLEEM() 

        print "-Loading 2D test grid"
        self.load_test_grid([0,5,10,15,20,25,30], [0,5,10,15,20,25,30])
#        self.load_test_grid([15], [15])
        print "-Reconstruction using centroid algorithm"
        self.reconstruct_grid_centroid()
        print "-Reconstruction using 2D maximum-a-posteriori"
        self.reconstruct_grid_2D_MAP()
        print "-Reconstruction using 3D maximum-a-posteriori (DepthEmbedding model)"
        self.reconstruct_grid_3D_MAP_DE()
        print "-Reconstruction using 3D maximum-a-posteriori (DepthEmbedding + MLEE model)"
        self.reconstruct_grid_3D_MAP_MLEEM()

        print "Computing Bias and Variance"
        self.compute_bias_and_variance() 
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


