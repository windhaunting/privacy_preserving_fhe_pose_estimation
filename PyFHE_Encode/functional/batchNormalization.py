#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 21:51:42 2021

@author: fubao
"""


# batch normalization layer


import numpy as np
from crypto import crypto as c


class BatchNormalization:
    """
    A class used to do the batch normalization layer
    ...

    Attributes
    ----------
    HE : Pyfhel
        Pyfhel object, used to encode weights and bias
    N, C, H, W

    Methods
    -------
    __init__(self, HE, weights, x_stride, y_stride, bias=None)
        Constructor of the layer, bias is set to None if not provided.
    __call__(self, t)
        Execute che convolution operation on a batch of images, t, in the form
            [n_images, n_layers, y, x]
        using weights, biases and strides of the layer.
    """

    def __init__(self, HE):
        self.HE = HE

        
    def __call__(self, t):
        result = np.array([self.BatchNorm2D(image) for image in t])
        return result


    def BatchNorm2D(self, x):
        """
        # use method from 
        # ref: <<FHE-compatible Batch Normalization for Privacy Preserving Deep Learning>>
        Parameters
        ----------
        x : np.array( dtype=PyCtxt )
        Encrypted image to execute the convolution on, in the form
        [y, x]
        gamma : float
            
        beta : float
            .

        Returns
        -------
        out : np.array( dtype=PyCtxt )
            DESCRIPTION.
        cache : tuple
            DESCRIPTION.

        """
        # 
        # we take variance from the freshly trained DNN and compute the inverse of
        #its square root over plaintext values. This newly transformed parameter denoted
        #by  is stored to be used in encrypted inference.
        eps=1e-5
        gamma = 1.0
        beta = 0.1 
        decrypt_x = c.decrypt_matrix(self.HE, x)
        
        sample_var = decrypt_x.var(axis=0)
        std = np.sqrt(sample_var + eps)
        
        v = gamma/std
        
        sample_mean = decrypt_x.mean(axis=0)

        tau = beta - sample_mean * v   # beta - np.dot(sample_mean, v)
        
        encry_v = c.encrypt_matrix(self.HE, v)
        encry_tau = c.encrypt_matrix(self.HE, tau)
        
        out = encry_v * x + encry_tau
        return out # , cache