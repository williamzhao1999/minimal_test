import warnings; warnings.simplefilter('ignore')  # hide warnings
from matplotlib import pyplot as plt
import numpy as np
from particles import distributions as dists
from particles import state_space_models as ssm
import json
from particles import mcmc
from particles import datasets as dts  # real datasets available in the package

f = open('A_last.json')
A_matrix = np.array(json.load(f))
f.close()

f = open('B.json')
B_matrix = np.array(json.load(f)[-1])
f.close()

f = open('H.json')
H_matrix = np.array(json.load(f)[-1])
f.close()

print(f"A matrix shape: {A_matrix.shape}, B matrix shape: {B_matrix.shape}, H matrix shape: {H_matrix.shape}")
a_matrix_shape = A_matrix.shape
b_matrix_shape = B_matrix.shape
h_matrix_shape = H_matrix.shape
paths = 24

keys = []
prior_dict = {}

for i in range(24):
    prior_dict['lambda'+str(i)] = dists.Gamma()
    keys.append('lambda'+str(i))
my_prior = dists.StructDist(prior_dict)

ccc = np.zeros(a_matrix_shape[0])
dd = np.eye(a_matrix_shape[0])

aa = dists.MvNormal(loc=np.zeros(a_matrix_shape[0]), cov=np.eye(a_matrix_shape[0])).rvs(size=1)

class Model(ssm.StateSpaceModel):
    def PX0(self):  # Distribution of X_0
        return dists.MvNormal(loc=np.zeros(a_matrix_shape[0]), cov=np.eye(a_matrix_shape[0]))
    def PX(self, t, xp):  # Distribution of X_t given X_{t-1}=xp (p=past)
        u = np.zeros(paths)
        for i in range(paths):
            u[i] = np.random.poisson(getattr(self, keys[i]), 1)[0]
        return dists.MvNormal(loc=np.matmul(A_matrix, xp) + np.matmul(B_matrix, u), cov=(0.5*np.eye(a_matrix_shape[0])))
    def PY(self, t, xp, x):  # Distribution of Y_t given X_t=x (and possibly X_{t-1}=xp)
        return dists.MvNormal(loc=np.matmul(H_matrix, x), cov=(0.5*np.eye(h_matrix_shape[0])))


# real data
T = 721
f = open('densities.json')
data = np.array(json.load(f))
f.close()

print(data.shape)

my_pmmh = mcmc.PMMH(ssm_cls=Model, prior=my_prior, data=data, Nx=456,
                   niter=1000)
my_pmmh.run(); 