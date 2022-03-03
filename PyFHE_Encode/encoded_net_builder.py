

#sys.path.append('/.')

#import convolutional_layer
#from functional.convolutional_layer import ConvolutionalLayer
import sys
import os


sys.path.append('../PyFHE_Encode/')
print ("current path: ", os.getcwd())
from functional.convolutional_layer import ConvolutionalLayer
from functional.square_layer import SquareLayer
from functional.average_pool import AveragePoolLayer
from functional.flatten_layer import FlattenLayer
from functional.rencryption_layer import RencryptionLayer
from functional.linear_layer import LinearLayer
from functional.ReLU import ReLU
from functional.batchNormalization import BatchNormalization


def build_from_pytorch(HE, net):
    """Given a PyTorch sequential net in a .pt/.pth file, returns
    an ordered list of encoded layers on which is possible
    to apply an encrypted computation.

    Parameters
    ----------
    HE : Pyfhel
        Pyfhel object
    net: nn.Sequential
        PyTorch model in nn.Sequential form

    Returns
    -------
    encoded_layers: list
        Ordered list of encoded layers which reflects
        the PyTorch model given in input
    """

    # Define builders for every possible layer

    def conv_layer(layer):
        if layer.bias is None:
            bias = None
        else:
            bias = layer.cpu().bias.detach().numpy()  # layer.bias.detach().numpy()

        return ConvolutionalLayer(HE, weights=layer.cpu().weight.detach().numpy(), # weights=layer.weight.detach().numpy()
                                  stride=layer.stride,
                                  padding=layer.padding,
                                  bias=bias)

    def lin_layer(layer):
        if layer.bias is None:
            bias = None
        else:
            bias = layer.bias.detach().numpy()
        return LinearLayer(HE, layer.weight.detach().numpy(),
                           bias)

    def avg_pool_layer(layer):
        # This proxy is required because in PyTorch an AvgPool2d can have kernel_size, stride and padding either of
        # type (int, int) or int, unlike in Conv2d
        kernel_size = (layer.kernel_size, layer.kernel_size) if isinstance(layer.kernel_size, int) else layer.kernel_size
        stride = (layer.stride, layer.stride) if isinstance(layer.stride, int) else layer.stride
        padding = (layer.padding, layer.padding) if isinstance(layer.padding, int) else layer.padding

        return AveragePoolLayer(HE, kernel_size, stride, padding)


    def max_pool_layer(layer):
        # max_pool_layer use the average pooling layer to replace it.
        kernel_size = (layer.kernel_size, layer.kernel_size) if isinstance(layer.kernel_size, int) else layer.kernel_size
        stride = (layer.stride, layer.stride) if isinstance(layer.stride, int) else layer.stride
        padding = (layer.padding, layer.padding) if isinstance(layer.padding, int) else layer.padding

        return AveragePoolLayer(HE, kernel_size, stride, padding)


    def flatten_layer(layer):
        return FlattenLayer()


    def square_layer(layer):
        return SquareLayer(HE)
    
    def relu_act(layer):
        return ReLU(HE)
    
    def batch_normalizaton(layer):
        return BatchNormalization(HE)
    
    # Maps every PyTorch layer type to the correct builder
    options = {"Conv": conv_layer,
               "Line": lin_layer,
               "Flat": flatten_layer,
               "AvgP": avg_pool_layer,
               "Squa": square_layer,
               "MaxP":max_pool_layer,
               "ReLU":relu_act,
               "Batc": batch_normalizaton
               }

    encoded_layers = [options[str(layer)[0:4]](layer) for layer in net]
    return encoded_layers
