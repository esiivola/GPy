from .kern import CombinationKernel
import numpy as np
from paramz.caching import Cache_this

# A thin wrapper around the base kernel to tell that we are dealing with a partial derivative of a Kernel
class DiffKern(CombinationKernel):
    def __init__(self, base_kern, dimension):
        super(DiffKern, self).__init__([base_kern], 'diffKern')
        self.base_kern = base_kern
        self.dimension = dimension

    def parameters_changed(self):
        self.base_kern.parameters_changed()

    @Cache_this(limit=3, ignore_args=())
    def K(self, X, X2, dimX2 = None): #X in dimension self.dimension
        if dimX2 is None:
            dimX2 = self.dimension
        return self.base_kern.dK2_dXdX2(X,X2, self.dimension, dimX2) #[:, :, self.dimension, dimX2]
 
    @Cache_this(limit=3, ignore_args=())
    def Kdiag(self, X):
        return np.diag(self.base_kern.dK2_dXdX2(X,X, self.dimension, self.dimension)) #[ :, :, self.dimension, self.dimension])
    
    @Cache_this(limit=3, ignore_args=())
    def dK_dX_wrap(self, X, X2): #X in dimension self.dimension
        return self.base_kern.dK_dX(X,X2, self.dimension) #[:,:, self.dimension]
    
    def reset_gradients(self):
        self.base_kern.reset_gradients()
    
    @Cache_this(limit=3, ignore_args=())
    def dK_dX2_wrap(self, X, X2): #X in dimension self.dimension
        return self.base_kern.dK_dX2(X,X2, self.dimension) #[:, :, self.dimension]
    
    def update_gradients_full(self, dL_dK, X, X2=None, dimX2=None):
        if dimX2 is None:
            dimX2 = self.dimension
        gradients = self.base_kern.dgradients2_dXdX2(X,X2, self.dimension, dimX2)
        self.base_kern.gradient = [np.sum(dL_dK*gradient) for gradient in gradients]
        #self.base_kern.update_gradients_direct(*[np.sum(dL_dK*gradient) for gradient in gradients])
        
    def update_gradients_diag(self, dL_dK_diag, X, reset=True): #X in dimension self.dimension
        gradients = self.base_kern.dgradients2_dXdX2(X,X, self.dimension, self.dimension)
        self.base_kern.gradient = [np.sum(dL_dK*gradient) for gradient in gradients]
        #self.base_kern.update_gradients_direct(*[np.sum(dL_dK_diag*np.diag(gradient)) for gradient in gradients])
    
    def update_gradients_dK_dX(self, dL_dK, X, X2=None): #X in dimension self.dimension
        gradients = self.base_kern.dgradients_dX(X,X2, self.dimension)
        self.base_kern.gradient = [np.sum(dL_dK*gradient) for gradient in gradients]
        #self.base_kern.update_gradients_direct(*[np.sum(dL_dK*gradient) for gradient in gradients])
        
    def update_gradients_dK_dX2(self, dL_dK, X, X2=None): #X in dimension self.dimension
        gradients = self.base_kern.dgradients_dX2(X,X2, self.dimension)
        self.base_kern.gradient = [np.sum(dL_dK*gradient) for gradient in gradients]
        #self.base_kern.update_gradients_direct(*[np.sum(dL_dK*gradient) for gradient in gradients])

    def gradients_X(self, dL_dK, X, X2):
        tmp = self.base_kern.gradients_XX(dL_dK, X, X2)[:,:,:, self.dimension]
        return np.sum(tmp, axis=1)

    def gradients_X2(self, dL_dK, X, X2):
        tmp = self.base_kern.gradients_XX(dL_dK, X, X2)[:, :, self.dimension, :]
        return np.sum(tmp, axis=1)