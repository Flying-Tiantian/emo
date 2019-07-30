from .LeNet import LeNet_model
from .Mobilenet import Mobilenet_model, Mobilenet_160_model, Mobilenet_64_model
# from .git_Mobilenet import Mobilenet_model, Mobilenet_160_model, Mobilenet_64_model
from .Resnet import Resnet18_model, Resnet20_model, Resnet56_model, Resnet56_64_model, Resnet20_128_model
from .Vgg import VGGa_model, VGG16_model, VGG19_model
from .Inception import Inception_model
from .Shufflenet import ShuffleNet_model
from .Squeezenet import Squeezenet_model

__all__ = ['ShuffleNet_model', 'Squeezenet_model', 'Inception_model', 'Mobilenet_64_model', 'Mobilenet_160_model', 'LeNet_model', 'Mobilenet_model', 'Resnet18_model', 'Resnet20_model', 'Resnet20_128_model', 'Resnet56_model', 'Resnet56_64_model', 'VGGa_model', 'VGG16_model', 'VGG19_model']