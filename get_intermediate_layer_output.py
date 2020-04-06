import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon.model_zoo import vision
import multiprocessing
from mxnet.gluon.data.vision.datasets import ImageFolderDataset


# To extract the feature from flatten layer of resnet_18:
resnet = gluon.model_zoo.vision.resnet18_v2(pretrained=True, ctx=mx.cpu(), prefix='model_')
inputs = mx.sym.var('data')
out = resnet(inputs)
internals = out.get_internals()
print(internals.list_outputs())
outputs = internals['model_dense0_fwd_output'],
# Create SymbolBlock that shares parameters with resnet
feat_model = gluon.SymbolBlock(outputs, inputs, params=resnet.collect_params())
x = mx.nd.random.normal(shape=(16, 3, 224, 224))
print(feat_model(x))



