from .kern import Kern
from .multioutput_kern import MultioutputKern, ZeroKern
import numpy as np
from functools import reduce, partial

class MultioutputDiffKern(MultioutputKern):
    def __init__(self, kernels, cross_covariances={}, name='MultioutputDiffKern'):
        if not isinstance(kernels, list):
            self.single_kern = True
            self.kern = kernels
            kernels = [kernels]
        else:
            self.single_kern = False
            self.kern = kernels
        # The combination kernel ALLWAYS puts the extra dimension last.
        # Thus, the index dimension of this kernel is always the last dimension
        # after slicing. This is why the index_dim is just the last column:
        self.index_dim = -1
        super(MultioutputKern, self).__init__(kernels=kernels, extra_dims=[self.index_dim], name=name, link_parameters=False)

        nl = len(kernels)
         
        #build covariance structure
        covariance = [[None for i in range(nl)] for j in range(nl)]
        linked = []
        for i in range(0,nl):
            unique=True
            for j in range(0,nl):
                if kernels[i].name != 'diffKern' and (i==j or (kernels[i] is kernels[j])):
                    covariance[i][j] = {'kern': kernels[i], 'K': kernels[i].K, 'update_gradients_full': kernels[i].update_gradients_full, 'gradients_X': kernels[i].gradients_X}
                    if i>j:
                        unique=False
                elif cross_covariances.get((i,j)) is not None: #cross covariance is given
                    covariance[i][j] = cross_covariances.get((i,j))
                elif kernels[i].name == 'diffKern' and kernels[i].base_kern == kernels[j]: # one is derivative of other
                    covariance[i][j] = {'kern': kernels[j], 'K': kernels[i].dK_dX_wrap, 'update_gradients_full': kernels[i].update_gradients_dK_dX, 'gradients_X': kernels[i].gradients_X}
                    unique=False
                elif kernels[j].name == 'diffKern' and kernels[j].base_kern == kernels[i]: # one is derivative of other
                    covariance[i][j] = {'kern': kernels[i], 'K': kernels[j].dK_dX2_wrap, 'update_gradients_full': kernels[j].update_gradients_dK_dX2, 'gradients_X': kernels[j].gradients_X2}
                elif kernels[i].name == 'diffKern' and kernels[j].name == 'diffKern' and kernels[i].base_kern == kernels[j].base_kern: #both are partial derivatives
                    covariance[i][j] = {'kern': kernels[i].base_kern, 'K': partial(kernels[i].K, dimX2=kernels[j].dimension), 'update_gradients_full': partial(kernels[i].update_gradients_full, dimX2=kernels[j].dimension), 'gradients_X':None}
                    if i>j:
                        unique=False
                else: # zero matrix
                    kern = ZeroKern()
                    covariance[i][j] = {'kern': kern, 'K': kern.K, 'update_gradients_full': kern.update_gradients_full, 'gradients_X': kern.gradients_X}       
            if unique is True:
                linked.append(i)
        self.covariance = covariance
        self.link_parameters(*[kernels[i] for i in linked])