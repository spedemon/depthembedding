
# This software is part of Occiput - http://occiput.io 
# March 2016
# Stefano Pedemonte
# Martinos Center for Biomedical Imaging, MGH, Boston, MA


from sklearn import manifold 
import scipy
import scipy.io
from scipy import stats
from numpy import zeros, ones, sort, unravel_index, repeat, sum, where, squeeze, fliplr, flipud
from numpy import log, tile, float32, argsort, int32, histogram, linspace, round, exp, convolve, sqrt, mgrid
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import os 
import sys 
import pylab as pl
pl.ion()



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

    def _print_percentage(self, batch, n_batches):
        percent = int32(100.0*(1.0*batch+1.0)/n_batches)
        if hasattr(self, "_previous_percent"): 
            if percent == self._previous_percent:
                update = False
            else: 
                update = True
                self._previous_percent = percent.copy() 
        else: 
            self._previous_percent = percent.copy() 
            update = True
        sys.stdout.write('%2.1d%% '%(percent,))
        sys.stdout.flush() 

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
            logSumL = log(L.sum(0)+EPS) 
            
            if self.prior is not None: 
                logPrior = log(self.prior)
                # if prior on depth (not on x,y,z), then repeat for each x,y location
                if len(logPrior.shape) == 1: 
                    logPrior = tile(logPrior,(Nx,Ny,1))
            else: 
                logPrior = 0 

            n_batches = N / batch_size + 1
            X = zeros(N) 
            Y = zeros(N) 
            Z = zeros(N)
            Energy = zeros(N)
            LogPosterior = zeros(N)
            Indexes = zeros(N)

            print ""
            for i_batch in range(n_batches): 
                self._print_percentage(i_batch, n_batches)
                if i_batch == n_batches-1: #last batch has arbitrary size
                     current_batch_size = N - batch_size * (n_batches-1)
                else: 
                     current_batch_size = batch_size
                index_start = i_batch * batch_size
                index_end   = index_start + current_batch_size
                D = data[index_start:index_end,:]
                sumD = D.sum(1) 
                logD = tile(logSumL,(current_batch_size,1)) * tile(sumD,(Nx*Ny*Nz,1)).T
                logposterior = D.dot(logL) - logD + tile(logPrior.reshape(((1,Nx*Ny*Nz))),(current_batch_size,1))
                index = int32(argsort(logposterior,axis=1))
                I = index[:,-1]
                #print logposterior.shape, I.shape, logposterior[range(len(I)),I].shape
                [x,y,z] = unravel_index(I,[Nx,Ny,Nz])
                X[index_start:index_end] = x
                Y[index_start:index_end] = y
                Z[index_start:index_end] = z 
                suml = self.forward_model[x,y,z,:].sum(1)
                Energy[index_start:index_end] = sumD/suml
                #Indexes[i_batch*batch_size:(i_batch+1)*batch_size] = I
                LogPosterior[index_start:index_end] = logposterior[range(len(I)),I]
            print ""
            return X, Y, Z, Energy, LogPosterior
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
            LogPosterior = zeros(N)
            Indexes = zeros(N)

            print ""
            for i_batch in range(n_batches): 
                self._print_percentage(i_batch, n_batches)
                if i_batch == n_batches-1: #last batch has arbitrary size
                     current_batch_size = N - batch_size * (n_batches-1)
                     index_end   = (i_batch+1)* current_batch_size
                else: 
                     current_batch_size = batch_size
                index_start = i_batch * batch_size
                index_end   = index_start + current_batch_size
                D = data[index_start:index_end,:]
                sumD = D.sum(1)
                #print "sumD:                         ", sumD.shape
                #print "tile(sumD,(Nx*Ny,1)).T:       ", tile(sumD,(Nx*Ny,1)).T.shape
                #print "logSumL:                      ", logSumL.shape
                #print "tile(logSumL,(current_batch_size,1)): ", tile(logSumL,(current_batch_size,1)).shape
                
                logD = tile(logSumL,(current_batch_size,1)) * tile(sumD,(Nx*Ny,1)).T
                logposterior = D.dot(logL) - logD + logPrior
                index = int32(argsort(logposterior,axis=1))
                I = index[:,-1] 
                [x,y] = unravel_index(I,[Nx,Ny])
                X[index_start:index_end] = x
                Y[index_start:index_end] = y
                suml = self.forward_model[x,y,:].sum(1)
                Energy[index_start:index_end] = (sumD+EPS)/(suml+EPS)
                #Indexes[index_start:index_end] = I
                LogPosterior[index_start:index_end] = logposterior[range(len(I)),I]
            print ""
            return X, Y, Energy, LogPosterior



def model_error(true_model, estimated_model, EPS=0.05, threshold_mean=0.5):
    true_model = true_model.flatten()
    estimated_model = estimated_model.flatten()
    error = ( abs( (true_model - estimated_model + EPS)/(true_model + EPS) ) )
    error = 100 * error[ true_model > threshold_mean * true_model.mean() ].mean()
    return error 



class ReconstructorCentroid():
    """Centroid algorithm for the 2D estimation of the coordinates of interaction 
    of a gamma photon.""" 
    def __init__(self, x_detectors, y_detectors, x_max, y_max, x_min=0.0, y_min=0.0, scale=1.0, shiftx=0.0, shifty=0.0, exponent=1.0): 
        self.exponent = exponent
        self.x_detectors = x_detectors
        self.y_detectors = y_detectors
        self.shiftx = shiftx
        self.shifty = shifty
        self.scale = scale
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def reconstruct(self, data, round=False): 
        data_size = data.shape[1]
        N = data.shape[0]
        X_detectors = tile(self.x_detectors, (N,1))
        Y_detectors = tile(self.y_detectors, (N,1)) 
        Energy = data.sum(1)
        X = (data**self.exponent * X_detectors).sum(1) / Energy
        Y = (data**self.exponent * Y_detectors).sum(1) / Energy
        X = (X + self.shiftx)*self.scale
        Y = (Y + self.shifty)*self.scale
        if round: 
            X = round(X)
            Y = round(Y)
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

    def sample(self, n_samples): 
        P = stats.rv_discrete(name='BeerLambert', values=(range(self.n_bins), self.get_probability_mass_function() )) 
        return P.rvs(size=n_samples)



class EnergySpectrum():
    """Data structure to compute the histogram of the energy of the gamma photons."""
    def __init__(self, n_bins=50, min_energy=0, max_energy=None, scale=None, peak=511.0): 
        """'min_energy' and 'max_energy' set the range; if not specified, they are set to the min and max of the data. 
        'scale' pre-scales the energy. If 'scale' is set to 'auto', 
        then the energy is scaled automatically to set the peak of the energy spectrum to 'peak'."""
        self.histogram = None
        self.energy = None
        self.min_energy = min_energy
        self.max_energy = max_energy 
        self.n_bins = n_bins 
        self.peak = peak
        self.scale = scale 

    def add_data(self, energies): 
        if self.scale is not None: 
            if self.scale is not "auto":
                energies = energies*self.scale                
        if self.histogram is None: 
            if self.scale is "auto": 
                [hist, energy] = histogram(energies, bins=self.n_bins, range=(energies.min(),energies.max()))
                energy_max = energy[where(hist==hist.max())[0]][0]
                scale = self.peak/energy_max
                self.scale = scale 
                energies = energies*self.scale  
            if self.min_energy is None: 
                self.min_energy = energies.min()
            if self.max_energy is None: 
                self.max_energy = energies.max() 
            [hist, energy] = histogram(energies, bins=self.n_bins, range=(self.min_energy,self.max_energy))
            self.histogram = hist
            self.energy = energy
        else: 
            [hist, energy] = histogram(energies, bins=self.n_bins, range=(self.min_energy,self.max_energy))
            self.histogram = self.histogram + hist 
    
    def get_spectrum(self): 
        return self.histogram, self.energy

    def show(self, fig=None): 
        fig = pl.figure(fig)
        pl.plot(self.energy[0:-1], self.histogram)
        return fig





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
                self.histogram[int(coordinates[0][i]),int(coordinates[1][i])] += 1
        else: 
            for i in range(len(coordinates[0])): 
                self.histogram[int(coordinates[0][i]),int(coordinates[1][i]),int(coordinates[2][i])] += 1        

    def get_histogram(self): 
        return self.histogram 

    def show(self, axis=2, index=None, interpolation='nearest', fig=None, cmap='hot', vmax=None, vmin=None):
        """Display histogram using pl. If histogram is 3D, it displays the plane perpendicular to 'axis'; 
        'index' indicates the plane index to display; if 'index' is None, it displays the integral of the histogram 
        in the direction of 'axis'."""
        if self.is_2D(): 
            image = self.get_histogram() 
        else: 
            if index is None: 
                image = self.get_histogram().sum(axis)
            else: 
                image = self.get_histogram()
                if axis==0: 
                    image = image[index,:,:]
                if axis==1: 
                    image = image[:,index,:]
                if axis==2: 
                    image = image[:,:,index]             
        fig = pl.figure(fig)
        if vmax is not None:
            if vmin is not None:   
                pl.imshow(image, interpolation=interpolation, cmap=cmap, vmax=vmax, vmin=vmin)
            else:
                pl.imshow(image, interpolation=interpolation, cmap=cmap, vmax=vmax)
        else: 
            if vmin is not None:   
                pl.imshow(image, interpolation=interpolation, cmap=cmap, vmin=vmin)
            else:
                pl.imshow(image, interpolation=interpolation, cmap=cmap)
        return fig





class LikelihoodFilter(): 
    """Filter events based on the likelihood value. """
    def __init__(self, forward_model_2D):
        self.forward_model_2D = forward_model_2D

    def filter(self, data, x, y, method="near_coordinates", points=2000, batch_size=0): 
        # MLE 2D
        reconstructor = ReconstructorMAP(self.forward_model_2D)
        [x_mle_2D, y_mle_2D, energy_mle_2D, logposterior_mle_2D] = reconstructor.reconstruct(data, batch_size=batch_size) 
        # Filter 
        if method=="match_coordinates": 
            indexes = where((x_mle_2D==x) & (y_mle_2D==y))
            inv_indexes = where(np.invert((x_mle_2D==x) & (y_mle_2D==y)))
        if method=="near_coordinates": 
            indexes = where((x_mle_2D>=x-1) & (x_mle_2D<=x+1) & (y_mle_2D>=y-1) & (y_mle_2D<=y+1))[0]
            inv_indexes = where(np.invert((x_mle_2D>=x-1) & (x_mle_2D<=x+1) & (y_mle_2D>=y-1) & (y_mle_2D<=y+1)))[0]
            if len(indexes)>=points:
                indexes = indexes[0:points]
        if method=='likelihood':
            logpind = np.flipud(int32(argsort(logposterior_mle_2D)))
            indexes = logpind[0:points]
            inv_indexes = logpind[points::]
            
        data_filtered = data[indexes,:] 
        data_leftout  = data[inv_indexes,:]
        return squeeze(data_filtered), squeeze(data_leftout)


class LikelihoodFilter3D(): 
    """Filter events based on the likelihood value. """
    def __init__(self, forward_model_3D, depth_prior):
        self.forward_model_3D = forward_model_3D
        self.depth_prior = depth_prior

    def filter(self, data, points=5000, batch_size=0): 
        # MLE 3D
        reconstructor = ReconstructorMAP(self.forward_model_3D)
        reconstructor.set_prior(self.depth_prior)
        [x_mle_3D, y_mle_3D, z_mle_3D, energy_mle_3D, logposterior_mle_3D] = reconstructor.reconstruct(data, batch_size=batch_size) 
        
        logpind = np.flipud(int32(argsort(logposterior_mle_3D)))
        indexes = logpind[0:points]
        inv_indexes = logpind[points::]
            
        data_filtered = data[indexes,:] 
        data_leftout  = data[inv_indexes,:]
        return squeeze(data_filtered), squeeze(data_leftout)
        



class ModelInterpolator():
    """Interpolation of the forward model (2D or 3D). Typically used for upsampling. """
    def __init__(self):
        pass 

    def _interp3(self, x,y,z, V, xi,yi,zi): 
        Xi,Yi,Zi = np.meshgrid(xi,yi,zi)
        C = zeros([Xi.size,3])
        C[:,0] = Xi.flatten()
        C[:,1] = Yi.flatten()
        C[:,2] = Zi.flatten()
        fn = RegularGridInterpolator((x,y,z), V)
        R = fn(C)
        R = R.reshape(Xi.shape)
        R = np.swapaxes(R,1,0)
        return R

    def _interp2(self, x,y, V, xi,yi):
        Xi,Yi = np.meshgrid(xi,yi)
        C = zeros([Xi.size,2])
        C[:,0] = Xi.flatten()
        C[:,1] = Yi.flatten()
        fn = RegularGridInterpolator((x,y), V)
        R = fn(C)
        R = R.reshape(Xi.shape)
        R = np.swapaxes(R,1,0)
        return R

    def resample(self, model, grid_size_interp = [200,200,60], interpolation="linear"): 
        if len(model.shape)==3:  
            data_dim = model.shape[2]
            grid_in_x = np.linspace(0.0, model.shape[0]-1, model.shape[0])
            grid_in_y = np.linspace(0.0, model.shape[1]-1, model.shape[1])
            
            grid_interp_x = np.linspace(0.0, model.shape[0]-1, grid_size_interp[0])
            grid_interp_y = np.linspace(0.0, model.shape[1]-1, grid_size_interp[1])

            model_interp = zeros((grid_size_interp[0],grid_size_interp[1],data_dim))
            for dim in range(data_dim): 
                m = self._interp2(grid_in_x, grid_in_y, model[:,:,dim].squeeze(), grid_interp_x, grid_interp_y)
                model_interp[:,:,dim] = m.reshape((grid_size_interp[0],grid_size_interp[1]))

        elif len(model.shape)==4:  #3D model
            data_dim = model.shape[3]
            x = np.linspace(0, model.shape[0]-1, model.shape[0])
            y = np.linspace(0, model.shape[1]-1, model.shape[1])
            z = np.linspace(0, model.shape[2]-1, model.shape[2])
            
            xi = np.linspace(0.0, model.shape[0]-1, grid_size_interp[0])
            yi = np.linspace(0.0, model.shape[1]-1, grid_size_interp[1])
            zi = np.linspace(0.0, model.shape[2]-1, grid_size_interp[2])

            model_interp = zeros((grid_size_interp[0],grid_size_interp[1],grid_size_interp[2],data_dim))
            for dim in range(data_dim): 
                V =  model[:,:,:,dim].squeeze()
                m = self._interp3(x, y, z, V, xi, yi, zi)
                model_interp[:,:,:,dim] = m.reshape((grid_size_interp[0],grid_size_interp[1],grid_size_interp[2]))
        else: 
            print "ModelInterpolator: invalid model - model must be 2D or 3D."
            model_interp = None
        return model_interp





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
            print "~Model2D: Calibration data does not have the correct size. "
            return 
        self.calibration_data = data 
        self.n_training  = data.shape[0]
        self.n_detectors = data.shape[1]

    def estimate_forward_model(self, allow_negative_values = False): 
        forward_model = zeros([1,self.n_detectors])
        for i in range(self.n_training): 
            forward_model += self.calibration_data[i,:] 
        forward_model /= self.n_training 
        # Don't allow negative values 
        if not allow_negative_values: 
            forward_model[forward_model<EPS]=EPS 
        self.forward_model = forward_model
        return self.forward_model

    def _check_data(self, data): 
        return len(data.shape)==2
        
    def visualize_model(self, x, y, reshape=(8,8), fig=None): 
        m = self.forward_model[x,y,:].reshape(reshape)
        fig = pl.figure(fig)
        pl.imshow(m)
        return fig



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
        prior = BeerLambert(alpha_inv_cm=0.83, n_bins=self.nz, bin_size_cm=1.0/self.nz) 
        self.set_depth_prior(prior.get_probability_mass_function() )
        
    def set_depth_prior(self, prior): 
        if len(prior) != self.nz: 
            print "~ModelDepthEmbedding: Not using specified depth prior (length %d). Length of depth prior vector must be equal to self.nz (%d)"%(len(prior),self.nz)
            return 
        self.prior = prior 

    def set_calibration_data(self, data):
        if not self._check_data(data): 
            print "~ModelDepthEmbedding: Calibration data does not have the correct size. "
            return 
        self.calibration_data = data 
        self.n_training  = data.shape[0]
        self.n_detectors = data.shape[1]
        
    def estimate_forward_model(self, allow_negative_values=False, unit_norm=False, zero_mean=False): 
        data = self.calibration_data.copy()
        N = data.shape[0]
        D = data.shape[1] 
        if zero_mean: 
            data = data - np.repeat(data.mean(1).reshape([N,1]),D,axis=1)
        if unit_norm: 
            data = data / np.repeat(np.linalg.norm(data,2,1).reshape([N,1]),D,axis=1)
        forward_model = zeros([self.nz,self.n_detectors]) 
        cumulative_prior = get_cumulative_prior(self.prior)

        #print "1-Projecting calibration data onto the manifold.."
        self.manifold = manifold.LocallyLinearEmbedding(self.n_neighbours, n_components=self.n_components, method=self.lle_method) 
        data_1d = self.manifold.fit_transform(data)[:,0]

        #print "2-Sorting the manifold.."
        data_1d = data_1d-data_1d.min()
        data_1d = data_1d/data_1d.max()
        data_1d = data_1d-data_1d.mean()
        data_1d_s = sort(data_1d)
        s = argsort(data_1d) 

        #print "3-Determining the orientation of the manifold.."
        distance_points = self.manifold.nbrs_.kneighbors(data)[0].sum(1)
        manifold_density = 1.0 / convolve(sqrt(distance_points[s]),ones([5*self.n_neighbours]), mode='valid') 
        L = len(manifold_density)
        if unit_norm:  # If normalizing the norm of the data vectors, the manifold density is inverted 
            if (manifold_density[0:L/2].mean() >= manifold_density[-L/2::].mean()): 
                invert = 1 
                data_1d = -data_1d
                data_1d_s = -flipud(data_1d_s)
                s = flipud(s) 
            else: 
                invert = 0 
        else:
            if (manifold_density[0:L/2].mean() < manifold_density[-L/2::].mean()): 
                invert = 1 
                data_1d = -data_1d
                data_1d_s = -flipud(data_1d_s)
                s = flipud(s) 
            else: 
                invert = 0 

        #print "4-Depth mapping.."   
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

        #print "5-Averaging measurement vectors.."
        data = self.calibration_data.copy()
        N_points_per_bin = histogram(membership,bins=self.nz)[0]
        for i in range(N): 
            z = membership[i]
            forward_model[z,:] += data[i,:]
        for z in range(self.nz): 
            forward_model[z,:] /= N_points_per_bin[z]

        self.membership = membership
        self.data_1d = data_1d
        self.data_1d_s = data_1d_s
        self.s = s 
        self.invert = invert
        self.manifold_density = manifold_density
        
        # Don't allow negative values 
        if not allow_negative_values: 
            forward_model[forward_model<EPS]=EPS 
        self.forward_model = forward_model
        return forward_model 

    def visualize_manifold(self, nd=3, membership=None, fig=None, unit_norm=False, zero_mean=False): 
        data = self.calibration_data.copy()
        N = data.shape[0]
        D = data.shape[1] 
        if zero_mean: 
            data = data - np.repeat(data.mean(1).reshape([N,1]),D,axis=1)
        if unit_norm: 
            data = data / np.repeat(np.linalg.norm(data,2,1).reshape([N,1]),D,axis=1)

        man = manifold.LocallyLinearEmbedding(self.n_neighbours, n_components=nd, method=self.lle_method) 
        data_nd = man.fit_transform(data) 
        
        if membership is None: 
            try: 
                membership = self.membership
            except: 
                membership=None
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        fig = plt.figure(fig)
        if nd==3: 
            ax = fig.add_subplot(111, projection='3d')
        else: 
            ax = fig.add_subplot(111)

        if membership is not None: 
            color = zeros([self.nz,3]) 
            color[:,1]=flipud(linspace(0,1,self.nz)) 
            for i in range(self.nz):
                if i%2:
                    color[i,0]=0; color[i,2]=1;
                else: 
                    color[i,0]=1; color[i,2]=0;

            cmap = zeros((N,3))
            for i in range(N): 
                cmap[i,:] = color[membership[i],:]
            if nd==1: 
                ax.scatter(data_nd[:,0], zeros(data_nd[:,0].shape), c=cmap, s=4, marker='.', edgecolors=cmap)
            elif nd==2: 
                ax.scatter(data_nd[:,0], data_nd[:,1], c=cmap, s=4, marker='.', edgecolors=cmap)
            else: 
                ax.scatter(data_nd[:,0], data_nd[:,1], data_nd[:,2], c=cmap, s=4, marker='.', edgecolors=cmap)
        else:
            if nd==1: 
                ax.scatter(data_nd[:,0], zeros(data_nd[:,0].shape), s=4, marker='.')
            elif nd==2: 
                ax.scatter(data_nd[:,0], data_nd[:,1], s=4, marker='.')
            else: 
                ax.scatter(data_nd[:,0], data_nd[:,1], data_nd[:,2], s=4, marker='.')
        ax.set_xlabel('r1')
        if nd>=2: 
            ax.set_ylabel('r2')
        if nd>=3:
            ax.set_zlabel('r3')
        plt.show()
        return fig

    def visualize_model(self, depth=None, reshape=(8,8), fig=None): 
        if depth is None: 
            m = self.forward_model[:,:].mean(0).reshape(reshape)
        else: 
            m = self.forward_model[depth,:].reshape(reshape)
        fig = pl.figure(fig)
        pl.imshow(m, resample='nearest')
        return fig 

    def _check_data(self, data): 
        return len(data.shape)==2



class ModelMLEEM(): 
    """Algorithm for the estimation of the characteristics of a gamma camera based on 
    Expectation Maximization Maximum-likelihood-estimation."""
    def __init__(self, initial_model):
        self.calibration_data = None 
        self.prior = None
        self.set_initial_model(initial_model)

    def estimate_forward_model(self, method="hard_plus_smoothing", n_max_iterations=15, smoothness=0.2): 
        nz = self.initial_model.shape[0]
        n_detectors = self.initial_model.shape[1]
        forward_model = self.initial_model.copy()
        if method is "hard": 
            for i in range(n_max_iterations): 
                reconstructor = ReconstructorMAP(forward_model) 
                reconstructor.set_prior(self.prior) 
                [xrec,yrec,zrec,energyrec,posterior] = reconstructor.reconstruct(self.calibration_data) 
                for z in range(nz): 
                    indexes = where(zrec==z)[0]
                    forward_model[0,0,z,:] = self.calibration_data[indexes,:].sum(0)/len(indexes)
        elif method is "soft": 
            for i in range(n_max_iterations): 
                pass # FIXME-implement
        elif method is "hard_plus_smoothing":
            for i in range(n_max_iterations): 
                reconstructor = ReconstructorMAP(forward_model) 
                reconstructor.set_prior(self.prior) 
                [xrec,yrec,zrec,energyrec,posterior] = reconstructor.reconstruct(self.calibration_data) 
                for z in range(nz): 
                    indexes = where(zrec==z)[0]
                    forward_model[0,0,z,:] = self.calibration_data[indexes,:].sum(0)/len(indexes)  
                forward_model = scipy.ndimage.filters.gaussian_filter1d(forward_model, sigma=smoothness, mode='reflect',axis=2)
        self.forward_model = forward_model 
        self.xrec = xrec
        self.yrec = yrec 
        self.zrec = zrec
        return forward_model

    def set_calibration_data(self, calibration_data): 
        self.calibration_data = calibration_data 

    def set_initial_model(self, initial_model): 
        if self.prior != None: 
            if initial_model.shape[0] != len(self.prior): 
                 print "~The number of depth bins in the forward model must match the length of the multinomial prior (%d)."%len(self.prior)
                 return
        self.initial_model = initial_model.reshape((1,1,initial_model.shape[0],initial_model.shape[1]))

    def set_depth_prior(self, prior): 
        if len(prior) != self.initial_model.shape[2]: 
            print "~Length of depth prior vector must be equal to the number of depth bins in the forward model (%d)."%self.initial_model.shape[2]
            return 
        self.prior = prior 

    def visualize_model(self, x, y, depth=None, reshape=(8,8), fig=None): 
        if depth is None: 
            m = self.forward_model[x,y,:,:].mean(0).reshape(reshape)
        else: 
            m = self.forward_model[x,y,depth,:].reshape(reshape)
        fig = pl.figure(fig)
        pl.imshow(m)
        return fig






class StatsCoordinates(): 
    """Compute statistics of reconstructed coordinates of interaction - i.e. bias, std, errors."""
    def __init__(self, recostructed_coordinates, groundtruth_coordinates, coordinates_range, method_name="", scale = (1.0, 1.0, 1.0)):
        self.method_name = method_name 
        self.recostructed_coordinates = recostructed_coordinates
        self.groundtruth_coordinates = groundtruth_coordinates
        self.coordinates_range = coordinates_range 
        self.scale = scale 
        self.is_3D = (len(self.recostructed_coordinates[0][0])==3) 

        self.compute_stats() 

    def compute_stats(self):
        n_points_x = self.groundtruth_coordinates[0].shape[0]
        n_points_y = self.groundtruth_coordinates[0].shape[1]
        self.bias = zeros([n_points_x,n_points_y])
        self.std = zeros([n_points_x,n_points_y])
        self.error = zeros([n_points_x,n_points_y])
        for ix in range(n_points_x): 
            for iy in range(n_points_y): 
                groundtruth_X = self.groundtruth_coordinates[0][ix][iy]
                groundtruth_Y = self.groundtruth_coordinates[1][ix][iy]
                reconstructed = self.recostructed_coordinates[ix][iy]
                reconstructed_x = reconstructed[0]
                reconstructed_y = reconstructed[1]
                if len(reconstructed)==3: 
                    reconstructed_z = reconstructed[2] 
                    indexes = (reconstructed_z>self.coordinates_range[2][0]) & (reconstructed_z<self.coordinates_range[2][1])
                    reconstructed_x = reconstructed_x[indexes]
                    reconstructed_y = reconstructed_y[indexes]
                reconstructed_X = reconstructed_x.mean() 
                reconstructed_Y = reconstructed_y.mean() 
                #print self.method_name, ix,iy, " true: [%2.2f,%2.2f]  reconstructed: [%2.2f,%2.2f]"%(groundtruth_X,groundtruth_Y,reconstructed_X,reconstructed_Y)
                self.bias[ix,iy] = sqrt((self.scale[0]*(reconstructed_X-groundtruth_X))**2+(self.scale[1]*(reconstructed_Y-groundtruth_Y))**2)  
                self.std[ix,iy] = sqrt((1.0/len(reconstructed_x)) * ((self.scale[0]*(reconstructed_x-reconstructed_X))**2+(self.scale[1]*(reconstructed_y-reconstructed_Y))**2).sum());  
                self.error[ix,iy] = sqrt((1.0/len(reconstructed_x)) * ((self.scale[0]*(reconstructed_x-groundtruth_X))**2+(self.scale[1]*(reconstructed_y-groundtruth_Y))**2).sum()); 
        self.Bias = self.bias.mean()
        self.Std = self.std.mean()
        self.Error = self.error.mean()


    def print_summary(self): 
        print "===================================================="
        print "Bias %s:   mean: %2.2f   min: %2.2f   max: %2.2f "%(self.method_name, self.bias.mean(), self.bias.min(), self.bias.max()) 
        print "Std %s:    mean: %2.2f   min: %2.2f   max: %2.2f "%(self.method_name, self.std.mean(), self.std.min(), self.std.max()) 
        print "Error %s:  mean: %2.2f   min: %2.2f   max: %2.2f "%(self.method_name, self.error.mean(), self.error.min(), self.error.max()) 
        print "----------------------------------------------------"

    def plot_depth_density(self, fig=None): 
        n_points_x = len(self.groundtruth_coordinates)
        n_points_y = len(self.groundtruth_coordinates[0])
        if not self.is_3D: 
            print "StatsCoordinates: the coordinates are 2D, no depth information to display"
            return 
        n_points_per_bin = zeros(self.coordinates_range[2])
        for ix in range(n_points_x): 
            for iy in range(n_points_y): 
                reconstructed = self.recostructed_coordinates[ix][iy]
                reconstructed_z = reconstructed[2]
                n_points_per_bin += histogram(reconstructed_z,bins=self.coordinates_range[2])[0]
        fig = pl.figure()
        pl.plot(range(self.coordinates_range[2]), n_points_per_bin)
        return fig


def get_data_cmice_8mm_crystal(x,y,path="./data_ftp/cmice_data/20140508_ZA0082/test_data/"): 
    """cMiCe camera data acquired on regular (x,y) grid: load from file the calibration or test data for a given beam position. 
    Return a ndarray."""
    filename = path + "/ROW%s_COL%s.mat"%(str(x+1).zfill(2), str(y+1).zfill(2))
    data = scipy.io.loadmat(filename)['data']
    data[data<0] = 0
    return float32(data.T)


def get_data_cmice_15mm_crystal(x,y,path="./data_ftp/cmice_data/20090416_AA1003_cmice_data_15mm/"): 
    filename = path + "%0.3f"%(1013*x/1000.0)+"_"+"%0.3f"%(1013*y/1000.0)+".mat" 
    import h5py
    print filename
    data = np.array(h5py.File(filename)['data']).T
    #data = scipy.io.loadmat(filename)['data']
    data[data<0] = 0
    return float32(data.T)



def get_data_cmice_45_degrees(index, path="./data_ftp/cmice_data/20090416_AA1003_cmice_data_15mm_45_deg/"):
    """cMiCe: data acquired with 45 degrees beam."""
    filenames = os.listdir(path)
    if index>=len(filenames): 
        return None
    filename = path+filenames[index]
    data = scipy.io.loadmat(filename)['data']
    data[data<0] = 0
    return float32(data.T)


