import emcee
import numpy as np
import matplotlib.pyplot as plt
from corner import corner

plt.ion()

#These are the functions to compute the model. Final model is mod4 and the four fitted parameters are rho0, rs, b, and c

def beta(xaxis, rho0, rs, b):
    return rho0*((1+(xaxis/rs)**2)**(3*b/2))**(-1)

def mod4(xaxis, rho0, rs, b, c):
    return beta(xaxis, rho0, rs, b) + c

#This is the function to compute the likelihood, theta are the parameters (rho0, rs, b, c), xaxis, the data in x, profile the data in y, and std the errors

def lnlike(theta, xaxis, profile, std):
    return -0.5*np.sum((mod4(xaxis, *theta)-profile)**2/std**2)

#Define the priors

def lnprior(theta, priors):
    if (priors[0][0] < theta[0] < priors[0][1]) & (priors[1][0] < theta[1] < priors[1][1]) & (priors[2][0] < theta[2] < priors[2][1]) & (priors[3][0] < theta[3] < priors[3][1]):
        return 0.0
    else:
        return -np.inf

#Compute the likelihood in function of the priors
    
def lnprob(theta, priors, xaxis, profile, std):
    lp = lnprior(theta, priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, xaxis, profile, std)

#These are the data. jeje[0] data in x, jeje[1] data in y, jeje[3] the errors

jeje = np.load('tomcmc.npy')

#Parametrisation of the MCMC

ndim = 4 #number of parameters to fit
nwalkers = 2*ndim #numbers of chain in parallel
nsteps = 100000 #number of steps in each chain
ncut = int(0.1*nsteps) #percentage of elements to remove from the beginning of the chains (to ensure the convergence)

#Priors
priors = [[0.008, 0.04], [0.01, 10.], [0.1, 1.], [0.4965, 0.505]]

#Initial positions
pos0 = [0.01791221, 3.48155049, 0.4263093, 0.500]

#Initial positions for each chain (noise added for independency of the chains)
pos = [pos0 + np.random.randn(ndim)*0.005*pos0 for i in range(nwalkers)]

#Define MCMC object
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(priors, jeje[0], jeje[1], jeje[2]))

#Run MCMC
sampler.run_mcmc(pos, nsteps)

#Restructure the chain of parameters
samples = sampler.chain[:, ncut:, :].reshape((-1, ndim))

#This is useful only for the plots
labels = [r'${f_\mathrm{Q}}_0$', r'$r_\mathrm{s}$', r'$\beta$', r'c']
toto = samples[np.random.randint(len(samples), size=50000)]
corner(toto, labels=labels, plot_datapoints=False, color='C0')
#plt.savefig('plots/mcmc.png')
#np.save('toto.npy', toto)
