import numpy as np

#docu:
# https://pyfhel.readthedocs.io/en/latest/_autosummary/Pyfhel.Pyfhel.html

class ReLU:
    """
    A class used to represent a layer which performs the approximate encoding of ReLU on the
    encrypted data for FHE

    ...

    Notes
    -----
    Relu(x) = max(0, x) 
    => approximate as  relu(x) = x^2
    
    not quite accurate
    
    or more accurate => low degree of polynomial
    0.1249 + 0.5000X + 0.3729X^2 + 0.0410X^4 + 0.0016X^6

    => use the approximate of this paper:
        
    https://eprint.iacr.org/2019/591.pdf  ;  Simulating Homomorphic Evaluation of Deep
Learning Predictions
   
    or 
    => this paper:
        Precise Approximation of Convolutional Neural Networks for Homomorphically Encrypted Data
        
    Attributes
    ----------

    HE: Pyfhel
        Pyfhel object

    Methods
    -------
    __init__(self, HE)
        Constructor of the layer.
    __call__(self, image)
        Executes the square of the input matrix.
    """
    
    def __init__(self, HE):
        self.HE = HE
        self.c0 = HE.encryptFrac(0.1249)
        self.c1 = HE.encryptFrac(0.5)
        self.c2 = HE.encryptFrac(0.3729)
        self.c4 = HE.encryptFrac(0.0410)
        self.c6 = HE.encryptFrac(0.0016)

    def relinearization(self, enc):
        
        self.HE.relinKeyGen(20, 100)
        enc = ~ enc
        return enc
    
    def power4(self, x):
        # call self.HE.power(x, 3) not correct when decrypted
        #power2_encry = self.HE.power(x, 2)
        #power2_plain = self.HE.decryptFrac(power2_encry)
        plain_x = self.HE.decryptFrac(x)
        power4_encry = self.HE.encryptFrac(plain_x**4)
        
        return power4_encry

    def power6(self, x):
        # call self.HE.power(x, 3) not correct when decrypted
        #power2_encry = self.HE.power(x, 2)
        #power2_plain = self.HE.decryptFrac(power2_encry)
        plain_x = self.HE.decryptFrac(x)
        power6_encry = self.HE.encryptFrac(plain_x**6)
        
        return power6_encry
    
    def relu1_helper(self, one_element):
        """
        Execute the approximate of relu for FHE
        with x^2
        Parameters
        ----------
        HE : Pyfhel object
        one_element : encrypted float
    
        Returns
        -------
        result : encrypted result of approxmation
        
        """
        #print("one_element type: ", type(one_element))

        relu_one_element = self.HE.power(one_element, 2)  # self.HE.power(x,4)
        relu_one_element = self.relinearization(relu_one_element)
        return relu_one_element
        
    
    
    def relu1(self, x):
        """

        Parameters
        ----------
        x : numpy array

        Returns
        -------
        output

        """
        """
        print("x type: ", type(x), x.shape)
        #eturn np.array(list(map(lambda one_element: self.relu1_helper(one_element), x)))
                                                        for third in first]

                            for third in first]
                            for second in first]
                           for first in x]
        """
        vfunc = np.vectorize(self.relu1_helper)
        return vfunc(x)
        
        
    def relu2_helper(self, one_element):
        """
        Execute the approximate of relu for FHE
        with lwo degree of oolynomial
        Parameters
        ----------
        HE : Pyfhel object
        one_element : encrypted data
    
        0.1249 + 0.5000X + 0.3729X^2 + 0.0410X^4 + 0.0016X^6

        Returns
        -------
        result : approximate relu encrypted result
        
        """

        tmp1 = self.HE.add(self.c0, self.HE.multiply(self.c1, one_element))
        tmp2 = self.HE.add(tmp1, self.HE.multiply(self.c2, self.HE.power(one_element,2)))
        tmp2 = self.relinearization(tmp2)
        tmp3 = self.HE.add(tmp2, self.HE.multiply(self.c4, self.power4(one_element)))  
        tmp3 = self.relinearization(tmp3)
        relu_one_element =  self.HE.add(tmp3, self.HE.multiply(self.c6, self.power6(one_element))) 
        relu_one_element = self.relinearization(relu_one_element)

        return relu_one_element

    def relu2(self, x):
        vfunc = np.vectorize(self.relu2_helper)
        return vfunc(x)


    def __call__(self, x):
        return self.relu1(x)   # self.relu1



    