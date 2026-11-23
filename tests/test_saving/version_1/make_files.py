"""Writes the version-1 test files: python save_main_small.py <dir> with
qewton of main (before structured dtypes) on PYTHONPATH."""
import sys, os, torch, warnings; warnings.filterwarnings("ignore")
from qewton.algorithms.building_blocks import ReLU
from qewton.algorithms.dl_models.convolutions.cnn import CNN, UNet
from qewton.algorithms.dl_models.convolutions.encoding import ConvolutionalEncoder
from qewton.algorithms.dl_models.fcn import FCN, DeepRitzNet
from qewton.algorithms.dl_models.harmonic_fcn import HarmonicFCN
from qewton.algorithms.dl_models.pca_net import PCANet
from qewton.config.variables import Variable
from qewton.config.saving.callables import save
out = sys.argv[1]
def su(m): m.setup(); return m
def pca():
    m = PCANet(input_variable=Variable("x", dim=3), output_variable=Variable("o", dim=1), pca_n_input=3, pca_n_output=1, data_source_node=None, fcn_hidden_layers=1, fcn_hidden_neurons=2)
    m.fit([torch.rand(20, 3)], [torch.rand(20, 1)]); return m
MODELS = {
 "fcn": (lambda: FCN(in_neurons=2, hidden_neurons=3, out_neurons=1, n_hidden_layers=1), (3, 2)),
 "fcn_variables": (lambda: FCN(in_neurons=Variable("x", 2), hidden_neurons=3, out_neurons=Variable("u", 1), n_hidden_layers=1), (3, 2)),
 "deep_ritz": (lambda: DeepRitzNet(in_neurons=2, width=3, out_neurons=1, depth=1), (3, 2)),
 "harmonic_fcn": (lambda: su(HarmonicFCN(input_dim=2, hidden_neurons=3, output_dim=1, n_hidden_layers=1, max_harmonic=2, bias=True, activation=ReLU)), (3, 2)),
 "cnn": (lambda: su(CNN(in_channels=2, hidden_channels=2, out_channels=1, n_hidden_layers=1, kernel_size=(3, 3))), (1, 2, 5, 5)),
 "unet": (lambda: su(UNet(in_channels=1, channels=(2, 4), out_channels=1, conv_kernel_size=(3, 3))), (1, 1, 8, 8)),
 "encoding": (lambda: su(ConvolutionalEncoder(in_channels=1, channels=(2,), out_channels=1, conv_kernel_size=(3, 3), input_shape=(1, 8, 8), fcn_hidden_neurons=2, fcn_hidden_layers=1)), (2, 1, 8, 8)),
 "pca_net": (pca, (4, 3)),
}
torch.manual_seed(0)
for name, (make, shape) in MODELS.items():
    model = make(); x = torch.rand(*shape)
    save(model, os.path.join(out, name), replace=True, compress=True)
    torch.save({"input": x, "output": model(x).detach()}, os.path.join(out, name, "expected.pt"))
