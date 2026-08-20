from .dl_models.fcn import FCN, DeepRitzNet
from .dl_models.convolutions.cnn import CNN, UNet
from .dl_models.convolutions.encoding import ConvolutionalEncoder
from .dl_models.harmonic_fcn import HarmonicEmbedding, HarmonicFCN

from .dl_models.pca_net import PCANet

from .dl_models.deeponet.base import DeepONet
from .dl_models.deeponet.merge_nodes import DefaultMerger
from .dl_models.deeponet.deeponet_fcn import FCNDeepONet
from .dl_models.deeponet.deeponet_cnn import CNNDeepONet
