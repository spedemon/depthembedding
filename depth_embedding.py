
# This software is part of Occiput
# March 2016
# Stefano Pedemonte
# Martinos Center for Biomedical Imaging, MGH, Boston, MA


from sklearn import manifold 
import scipy
import scipy.io
from numpy import zeros, ones




class ReconstructorMAP(): 
    """Maximum a-posteriori estimation of the coordinates of interaction and energy
    of a gamma photon (2D and 3D)."""
    def __init__(self, forward_model): 
        self.forward_model = forward_model 

    def reconstruct(self, data): 
        pass #FIXME



class Reconstructor_CenterOfMass():
    """Center of mass algorithm for the 2D estimation of the coordinates of interaction 
    of a gamma photon.""" 
    def __init__(self, exponent=1.0): 
        self.exponent = 1.0
    
    def reconstruct(self, data): 
        pass #FIXME




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






class DepthEmbedding(): 
    """DepthEmbedding algorithm for the estimation of the characteristics of a gamma camera."""
    def __init__(self, n_depth_bins=16, n_neighbours=12, n_components=1, lle_method='standard'): 
        self.n_depth_bins = n_depth_bins
        self.n_neighbours = n_neighbours
        self.n_components = n_components
        self.lle_method = lle_method
        self.calibration_data = None 
        self.n_training = 0
        self.n_detectors = 0 
        self.set_default_depth_prior() 
        
    def set_default_depth_prior(self): 
        bl = BeerLambert() 
        self.set_depth_prior(bl.get_probability_mass_function() )
        
    def set_depth_prior(self, prior): 
        if len(prior) != self.n_depth_bins: 
            print "Length of depth prior vector must be equal to self.n_depth_bins (%d)"%self.n_depth_bins
            return 
        self.prior = prior 

    def set_calibration_data(self, data):
        if not self._check_data(data): 
            print "Calibration data does not have the correct size. "
            return 
        self.calibration_data = data 
        self.n_training  = data.shape[0]
        self.n_detectors = data.shape[1]

    def estimate_forward_model(self, method="DepthEmbedding"): 
        if method is "2D": 
            forward_model = zeros([1,self.n_detectors])
            for i in range(self.n_training): 
                forward_model += self.calibration_data[i,:] 
                forward_model /= self.n_training 
        elif method is "DepthEmbedding": 
            forward_model = zeros([self.n_depth_bins,self.n_detectors]) 
            print "1-Projecting calibration data onto the manifold.."
            self.manifold = manifold.LocallyLinearEmbedding(self.n_neighbors, n_components=self.n_components, method=self.lle_method) 
            self.manifold.fit_transform(self.calibration_data) 
            print "2-Depth mapping.."
            
        else: 
            print "Unsupported method"
            return 
        return forward_model 

    def visualize_manifold(self): 
        pass #FIXME 

    def _check_data(self, data): 
        return len(data.shape)==2


    

class MLEEM(): 
    """Algorithm for the estimation of the characteristics of a gamma camera based on 
    Expectation Maximization Maximum-likelihood-estimation."""
    def __init__(self):
        self.caibration_data = None 

    def estimate_forward_model(self, initial_guess, method="hard", n_max_iterations=10, toll=1e-3): 
        forward_model = initial_guess 
        if method is "hard": 
            for i in range(n_max_iterations): 
                pass # FIXME
        elif method is "soft": 
            for i in range(n_max_iterations): 
                pass # FIXME
        return forward_model 

    def set_calibration_data(self, calibration_data): 
        self.calibration_data = calibration_data 




def get_data_cmice(x,y,type="calibration"): 
    """cMiCe camera data: load from file the calibration or test data for a given beam position. 
    Return a ndarray."""
    calibration_datapath="/Users/spedemon/Desktop/Experiments/DOI/DOI_Larry/data_ftp/cmice_data/20140508_ZA0082/test_data/"
    test_datapath="/Users/spedemon/Desktop/Experiments/DOI/DOI_Larry/data_ftp/cmice_data/20140508_ZA0082/test_data/"
    filename = "/ROW%s_COL%s.mat"%(str(x+1).zfill(2), str(y+1).zfill(2))
    if type is "calibration": 
        filename = calibration_datapath + filename
    elif type is "test": 
        filename = test_datapath + filename 
    data = scipy.io.loadmat(filename)
    data[data<0] = 0
    return data['data'].T
    



class DemoCmice(): 
    """Demo of estimation of the forward model of the cMiCe PET camera and reconstruction using various 2D and 3D methods. """
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
        self.forward_model_DE = None
        self.forward_model_MLEEM = None

    def _set_depth_prior(self): 
        prior_model = BeerLambert(self.scintillator_attenuation_coefficient, self.nz, self.scintillator_thickness/self.nz)
        self.depth_prior = prior_model.get_probability_mass_function()

    def estimate_forward_model_2D(self): 
        model = zeros([self.nx, self.ny, self.n_detectors])
        for ix in range(self.nx): 
            for iy in range(self.ny): 
                calibration_data = get_data_cmice(ix,iy)
                model_estimator = DepthEmbedding(n_depth_bins=self.nz, n_neighbours=self.n_neighbours)
                model_estimator.set_calibration_data(calibration_data)
                model_estimator.set_depth_prior(self.depth_prior)
                model[ix,iy,:] = model_estimator.estimate_forward_model("2D") 
        self.forward_model_2D = model 
        return self.forward_model_2D

    def estimate_forward_model_DE(self): 
        pass 
    
    def estimate_forward_model_MLEEM(self): 
        pass 
    
    def load_test_grid(self):
        pass 

    def reconstruct_grid_centroid(self): 
        pass 

    def reconstruct_grid_2D_MAP(self): 
        pass 

    def reconstruct_grid_3D_MAP_DE(self): 
        pass 

    def reconstruct_grid_3D_MAP_MLEEM(self): 
        pass 

    def visualize_results(self): 
        pass 

    def run(self): 
        print "-Estimating the forward model 2D .."
        self.estimate_forward_model_2D() 
        print "-Estimating the forward model DepthEmbedding .."
        self.estimate_forward_model_DE() 
        print "-Estimating the forward model MLEEM .."
        self.estimate_forward_model_MLEEM() 

        self.load_test_grid()
        self.reconstruct_grid_centroid()
        self.reconstruct_grid_2D_MAP()
        self.reconstruct_grid_3D_MAP_DE()
        self.reconstruct_grid_3D_MAP_MLEEM()

        self.visualize_results() 

    def enable_cache(self): 
        self._cache = True
    
    def disable_cache(self): 
        self._cache = False 




def get_data_simulation(x,y): 
    """Load from file the ground-truth forward model of a simulated monolithic gamma camera. """
    pass #FIXME



class DemoSimulation(DemoCmice): 
    """Demo of estimation of the forward model of a simulated monolithic PET camera 
    and reconstruction using various 2D and 3D methods. """
    def run(self): 
        pass 
    




if __name__ == "__main__": 
    print "--- DepthEmbedding ---"
    print "-- Running cMiCe PET camera calibration .."
    cmice = DemoCmice()
    cmice.run() 
    
    print "-- Running monolithic gamma camera simulation .."
    simulation = DemoSimulation()
    sumulation.run()
    
    print "-- Done"


