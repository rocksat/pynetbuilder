"""
Copyright 2017 Microsoft Corp.
Licensed under the terms of the 2 clause BSD license.
Please see LICENSE file in the project root for terms.
"""

from caffe.proto import caffe_pb2
import google.protobuf as pb
from caffe import layers as L
from caffe import params as P
import caffe
import sys
from argparse import ArgumentParser
sys.path.append('netbuilder')
from netbuilder.lego.hybrid import ConvBNReLULego, DWConvLego
from netbuilder.lego.base import BaseLegoFunction, Config
from netbuilder.lego.data import ImageDataLego
from netbuilder.tools.complexity import get_complexity

parser = ArgumentParser(description=""" This script generates imagenet alexnet train_val.prototxt files""")
parser.add_argument('-o', '--output_folder', help="""Train and Test prototxt will be generated as train.prototxt and test.prototxt""")


def write_prototxt(is_train, source, output_folder):
    netspec = caffe.NetSpec()
    if is_train:
        include = 'train'
        use_global_stats = False
        batch_size = 256
    else:
        include = 'test'
        use_global_stats = True
        batch_size = 100

    Config.del_default_params('Convolution', 'bias_filler')
    Config.set_default_params('Convolution', 'param', [dict(lr_mult=1, decay_mult=1)])

    # data layer
    params = dict(name='data', source=source, batch_size=batch_size, include=include,
                  image_data_param=dict(new_height=256, new_width=256),
                  transform_param=dict(crop_size=224, scale=0.017, mirror=False,
                  mean_value=[103.94, 116.78, 123.68]))
    data, label = ImageDataLego(params).attach(netspec)

    params = dict(name='1', num_output=32, kernel_size=3, pad=1, stride=2, bias_term=False,
                  use_global_stats=use_global_stats)
    conv1 = ConvBNReLULego(params).attach(netspec, [data])

    names   = ['2_1', '2_2', '3_1', '3_2', '4_1', '4_2', '5_1', '5_2', '5_3', '5_4', '5_5', '5_6', '6']
    groups  = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024]
    outputs = [64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]
    strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2]

    last = conv1
    for stage in range(13):
        params = dict(name=names[stage], group=groups[stage], num_output=outputs[stage],
                stride=strides[stage], use_global_stats=use_global_stats)
        last = DWConvLego(params).attach(netspec, [last])

    pool_params = dict(name='pool_6', pool=P.Pooling.AVE, global_pooling=True)
    pool6 = BaseLegoFunction('Pooling', pool_params).attach(netspec, [last])

    ip_params = dict(name='fc_7', num_output=1000, kernel_size=1)
    fc7 = BaseLegoFunction('Convolution', ip_params).attach(netspec, [pool6])

    smax_loss = BaseLegoFunction('SoftmaxWithLoss', dict(name='loss')).attach(netspec, [fc7, label])

    if include == 'test':
        BaseLegoFunction('Accuracy', dict(name='top1/acc', accuracy_param=dict(top_k=1))).attach(netspec, [fc7, label])
        BaseLegoFunction('Accuracy', dict(name='top5/acc', accuracy_param=dict(top_k=5))).attach(netspec, [fc7, label])
    filename = 'train.prototxt' if is_train else 'test.prototxt'
    filepath = output_folder + '/' + filename
    fp = open(filepath, 'w')
    print >> fp, netspec.to_proto()
    fp.close()

if __name__ == '__main__':
    args = parser.parse_args()
    write_prototxt(True, 'data/ilsvrc12/img_train.txt', args.output_folder)
    write_prototxt(False, 'data/ilsvrc12/img_val.txt', args.output_folder)

    # Also print out the network complexity
    # filepath = args.output_folder + '/test.prototxt'
    # params, flops = get_complexity(prototxt_file=filepath)
    # print 'Number of params: ', (1.0 * params) / 1000000.0, ' Million'
    # print 'Number of flops: ', (1.0 * flops) / 1000000.0, ' Million'
