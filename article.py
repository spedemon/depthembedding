
import cmice_simulation
import depth_embedding
import matplotlib.pyplot as plt
import numpy as np 


do_8mm  = False
do_15mm = False 
do_simulation = False

#########################################################################################
#########################################################################################
# Simulation of the cMiCE monolithic PET camera with 15mm scintillator crystal
#########################################################################################
#########################################################################################

if do_simulation: 
    import cmice_simulation

    ts1 = cmice_simulation.CmiceSimulation_15mm(nx_resample=49, ny_resample=49, nz_resample=24)
    ts1.run()

    ts2 = cmice_simulation.CmiceSimulation_15mm(nx_resample=98, ny_resample=98, nz_resample=48)
    ts2.run()

#########################################################################################
#########################################################################################
# cMiCE monolithic PET camera with 8mm scintillator crystal
#########################################################################################
#########################################################################################

if do_8mm: 
    import cmice_8mm

    te1 = cmice_8mm.Cmice_8mm(nx_resample=32, ny_resample=32, nz_resample=16)
    te1.run()

    te2 = cmice_8mm.Cmice_8mm(nx_resample=128, ny_resample=128, nz_resample=32)
    te2.run()

#########################################################################################
#########################################################################################
# cMiCE monolithic PET camera with 15mm scintillator crystal - including 45-degrees beam
#########################################################################################
#########################################################################################

if do_15mm:
    import cmice_15mm

    tf1 = cmice_15mm.Cmice_15mm(nx_resample=26, ny_resample=49, nz_resample=20)
    tf1.run()

    tf2 = cmice_15mm.Cmice_15mm(nx_resample=52, ny_resample=98, nz_resample=40)
    tf2.run()


#########################################################################################
#########################################################################################
# Make miscellaneous images 
#########################################################################################
#########################################################################################



##############################################################################
# Save images true forward model simulation cMiCE 15mm
##############################################################################
nz = 24
path = './results_cmice_sim_15mm/images/'
t = cmice_simulation.CmiceSimulation_15mm()

f = t.load_forward_model_true() 
f = scipy.ndimage.filters.gaussian_filter1d(f, sigma=1.5, mode='reflect',axis=2)

x = 2
y = 32

fig=plt.figure()
fig.set_size_inches(5,5)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

plt.axis('off')
for z in range(nz): 
    image = f[x,y,z,:].reshape([8,8])
    plt.imshow(image, aspect='equal', interpolation='nearest',vmax=100, cmap='hot') 
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    filename = path+'forward_model_%d_%d_%d.png'%(x,y,z)
    plt.savefig(filename, bbox_inches=extent)


##############################################################################
# Save images true forward model real data CMiCE 15mm
##############################################################################
nz = 20
path = './results_cmice_15mm/images/'

import cmice_15mm

t = cmice_15mm.Cmice_15mm(nx_resample=26, ny_resample=49, nz_resample=20)
f = t.load_forward_model_DE() 

x = 21
y = 34

fig=plt.figure()
fig.set_size_inches(5,5)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

plt.axis('off')
for z in range(nz): 
    image = f[x,y,z,:].reshape([8,8])
    plt.imshow(image, aspect='equal', interpolation='nearest',vmax=210, cmap='hot') 
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    filename = path+'forward_model_%d_%d_%d.png'%(x,y,z)
    plt.savefig(filename, bbox_inches=extent)
    
##############################################################################
# Save images reconstructed forward models 
##############################################################################
f = t.load_forward_model_DE() 

fig=plt.figure()
fig.set_size_inches(5,5)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

plt.axis('off')
for z in range(nz): 
    image = f[x,y,z,:].reshape([8,8])
    plt.imshow(image, aspect='equal', interpolation='nearest',vmax=100, cmap='hot') 
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    filename = path+'estim_model_DE_%d_%d_%d.png'%(x,y,z)
    plt.savefig(filename, bbox_inches=extent)



f = t.load_forward_model_MLEEM() 

fig=plt.figure()
fig.set_size_inches(5,5)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

plt.axis('off')
for z in range(nz): 
    image = f[x,y,z,:].reshape([8,8])
    plt.imshow(image, aspect='equal', interpolation='nearest',vmax=100, cmap='hot') 
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    filename = path+'estim_model_MLEEM_%d_%d_%d.png'%(x,y,z)
    plt.savefig(filename, bbox_inches=extent)



f = t.load_forward_model_2D() 

fig=plt.figure()
fig.set_size_inches(5,5)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
plt.axis('off')

image = f[x,y,:].reshape([8,8])
plt.imshow(image, aspect='equal', interpolation='nearest',vmax=100, cmap='hot') 
extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
filename = path+'estim_model_2D_%d_%d.png'%(x,y)
plt.savefig(filename, bbox_inches=extent)

##############################################################################
# Create images manifold 


##############################################################################
# Save images reconstructed grid 
##############################################################################




##############################################################################
# Evaluate error in model recovery with varying LLE neighbor size
##############################################################################

eta_range = np.int32(np.arange(4,40,1))
x          = 10 
y          = 40 
N          = 4000

true_model = f[x,y,:,:].squeeze()
data, depth = t.get_data_grid(x,y)
prior = depth_embedding.BeerLambert(n_bins=nz, bin_size_cm=1.5/nz).get_probability_mass_function()

error = np.zeros(eta_range.shape)

for i in range(len(eta_range)):
    eta = eta_range[i]
    
    print "DepthEmbedding, eta = %d"%eta
    D = depth_embedding.ModelDepthEmbedding(nz=nz, n_neighbours=eta)
    D.set_calibration_data(data[0:N,:])
    D.set_depth_prior(prior)

    model = D.estimate_forward_model() 
    error[i] = depth_embedding.model_error(true_model, model)

plt.figure()
plt.plot(eta_range, error, 'b') 
plt.axis([eta_range.min(), eta_range.max(), 0.0, 10.0])

##############################################################################
# Evaluate error in model recovery with varying number of training points
##############################################################################

N_range    = np.int32(np.arange(200,10000,200))
#N_range = np.int32([1500,2000,])

x          = 10 
y          = 20 

true_model = f[x,y,:,:].squeeze()
data, depth = t.get_data_grid(x,y)
prior = depth_embedding.BeerLambert(n_bins=nz, bin_size_cm=1.5/nz).get_probability_mass_function()

error_DE    = np.zeros(N_range.shape)
error_MLEEM = np.zeros(N_range.shape)

for i in range(len(N_range)):
    N = N_range[i]
    
    print "DepthEmbedding, N = %d"%N
    D = depth_embedding.ModelDepthEmbedding(nz=nz)
    D.set_calibration_data(data[0:N,:])
    D.set_depth_prior(prior)

    model_DE = D.estimate_forward_model() 
    error_DE[i] = depth_embedding.model_error(true_model, model_DE)
    
    print "MLEEM,          N = %d"%N
    M = depth_embedding.ModelMLEEM(model_DE)
    M.set_calibration_data(data[0:N,:])
    M.set_depth_prior(prior) 
    
    model_MLEEM = M.estimate_forward_model(n_max_iterations=100, smoothness=0.4) 
    error_MLEEM[i] = depth_embedding.model_error(true_model, model_MLEEM)

plt.figure()
plt.plot(N_range, error_DE, 'b') 
plt.hold(1)
plt.plot(N_range, error_MLEEM, 'r') 


