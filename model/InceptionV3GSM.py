from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import paddle
from paddle import ParamAttr
import paddle.nn as nn
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import Uniform
import math

from model.gsm import gsmModule
# from gsm import gsmModule

__all__ = ["InceptionV3"]


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 padding=0,
                 groups=1,
                 act="relu",
                 name=None):
        super(ConvBNLayer, self).__init__()

        self.conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=ParamAttr(name=name+"_weights"),
            bias_attr=False)
        self.batch_norm = BatchNorm(
            num_filters,
            act=act,
            param_attr=ParamAttr(name=name+"_bn_scale"),
            bias_attr=ParamAttr(name=name+"_bn_offset"),
            moving_mean_name=name+"_bn_mean",
            moving_variance_name=name+"_bn_variance")

    def forward(self, inputs):
        y = self.conv(inputs)
        y = self.batch_norm(y)
        return y

class InceptionStem(nn.Layer):
    def __init__(self, channels):
        super(InceptionStem, self).__init__()
        self.conv_1a_3x3 = ConvBNLayer(num_channels=channels,
                                       num_filters=32,
                                       filter_size=3,
                                       stride=2,
                                       act="relu",
                                       name="conv_1a_3x3")
        self.conv_2a_3x3 = ConvBNLayer(num_channels=32,
                                       num_filters=32,
                                       filter_size=3,
                                       stride=1,
                                       act="relu",
                                       name="conv_2a_3x3")
        self.conv_2b_3x3 = ConvBNLayer(num_channels=32,
                                       num_filters=64,
                                       filter_size=3,
                                       padding=1,
                                       act="relu",
                                       name="conv_2b_3x3")
        self.maxpool = MaxPool2D(kernel_size=3, stride=2, padding=0)
        self.conv_3b_1x1 = ConvBNLayer(num_channels=64,
                                       num_filters=80,
                                       filter_size=1,
                                       act="relu",
                                       name="conv_3b_1x1")        
        self.conv_4a_3x3 = ConvBNLayer(num_channels=80,
                                       num_filters=192,
                                       filter_size=3,
                                       act="relu",
                                       name="conv_4a_3x3")
    def forward(self, x):
        y = self.conv_1a_3x3(x)
        y = self.conv_2a_3x3(y)
        y = self.conv_2b_3x3(y)
        y = self.maxpool(y)
        y = self.conv_3b_1x1(y)
        y = self.conv_4a_3x3(y)
        y = self.maxpool(y)
        return y

                         
class InceptionA(nn.Layer):
    def __init__(self, num_channels, pool_features, name=None):
        super(InceptionA, self).__init__()
        self.branch1x1 = ConvBNLayer(num_channels=num_channels,
                                     num_filters=64,
                                     filter_size=1,
                                     act="relu",
                                     name="inception_a_branch1x1_"+name)

        self.gsm = gsmModule(fPlane=64, num_segments=16)

        self.branch5x5_1 = ConvBNLayer(num_channels=num_channels,
                                       num_filters=48, 
                                       filter_size=1, 
                                       act="relu",
                                       name="inception_a_branch5x5_1_"+name)
        self.branch5x5_2 = ConvBNLayer(num_channels=48, 
                                       num_filters=64, 
                                       filter_size=5, 
                                       padding=2, 
                                       act="relu",
                                       name="inception_a_branch5x5_2_"+name)

        self.branch3x3dbl_1 = ConvBNLayer(num_channels=num_channels,
                                       num_filters=64, 
                                       filter_size=1, 
                                       act="relu",
                                       name="inception_a_branch3x3dbl_1_"+name)
        self.branch3x3dbl_2 = ConvBNLayer(num_channels=64,
                                       num_filters=96, 
                                       filter_size=3, 
                                       padding=1,
                                       act="relu",
                                       name="inception_a_branch3x3dbl_2_"+name)
        self.branch3x3dbl_3 = ConvBNLayer(num_channels=96,
                               num_filters=96, 
                               filter_size=3, 
                               padding=1,
                               act="relu",
                               name="inception_a_branch3x3dbl_3_"+name)
        self.branch_pool = AvgPool2D(kernel_size=3, stride=1, padding=1, exclusive=False)
        self.branch_pool_conv = ConvBNLayer(num_channels=num_channels,
                               num_filters=pool_features, 
                               filter_size=1, 
                               act="relu",
                               name="inception_a_branch_pool_"+name)
    def forward(self, x):
        # print('before A.shape', x.shape)
        branch1x1 = self.branch1x1(x)
        # print('A.shape', branch1x1.shape)
        branch1x1 = self.gsm(branch1x1)
        # print('A gsm.shape', branch1x1.shape)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.branch_pool(x)

        branch_pool = self.branch_pool_conv(branch_pool)
        outputs = paddle.concat([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=1)
        # print('after A gsm.shape', outputs.shape)
        return outputs

    
class InceptionB(nn.Layer):
    def __init__(self, num_channels, name=None):
        super(InceptionB, self).__init__()
        
        self.branch3x3 = ConvBNLayer(num_channels=num_channels,
                                     num_filters=384,
                                     filter_size=3,
                                     stride=2,
                                     act="relu",
                                     name="inception_b_branch3x3_"+name) 
        self.branch3x3dbl_1 = ConvBNLayer(num_channels=num_channels, 
                                       num_filters=64, 
                                       filter_size=1, 
                                       act="relu",
                                       name="inception_b_branch3x3dbl_1_"+name)

        self.gsm = gsmModule(fPlane=64, num_segments=16)

        self.branch3x3dbl_2 = ConvBNLayer(num_channels=64, 
                                       num_filters=96, 
                                       filter_size=3, 
                                       padding=1,
                                       act="relu",
                                       name="inception_b_branch3x3dbl_2_"+name)
        self.branch3x3dbl_3 = ConvBNLayer(num_channels=96, 
                                       num_filters=96, 
                                       filter_size=3,
                                       stride=2,
                                       act="relu",
                                       name="inception_b_branch3x3dbl_3_"+name)
        self.branch_pool = MaxPool2D(kernel_size=3, stride=2)
        
    def forward(self, x):        
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.gsm(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.branch_pool(x)

        outputs = paddle.concat([branch3x3, branch3x3dbl, branch_pool], axis=1)

        return outputs

class InceptionC(nn.Layer):
    def __init__(self, num_channels, channels_7x7, name=None):
        super(InceptionC, self).__init__()
        self.branch1x1 = ConvBNLayer(num_channels=num_channels, 
                                       num_filters=192, 
                                       filter_size=1, 
                                       act="relu",
                                       name="inception_c_branch1x1_"+name)

        self.gsm = gsmModule(fPlane=192, num_segments=16)

        self.branch7x7_1 = ConvBNLayer(num_channels=num_channels, 
                                       num_filters=channels_7x7, 
                                       filter_size=1, 
                                       stride=1,
                                       act="relu",
                                       name="inception_c_branch7x7_1_"+name)
        self.branch7x7_2 = ConvBNLayer(num_channels=channels_7x7,
                                       num_filters=channels_7x7, 
                                       filter_size=(1, 7), 
                                       stride=1,
                                       padding=(0, 3),
                                       act="relu",
                                       name="inception_c_branch7x7_2_"+name)
        self.branch7x7_3 = ConvBNLayer(num_channels=channels_7x7,
                                       num_filters=192, 
                                       filter_size=(7, 1), 
                                       stride=1,
                                       padding=(3, 0),
                                       act="relu",
                                       name="inception_c_branch7x7_3_"+name)
        
        self.branch7x7dbl_1 = ConvBNLayer(num_channels=num_channels, 
                                       num_filters=channels_7x7, 
                                       filter_size=1, 
                                       act="relu",
                                       name="inception_c_branch7x7dbl_1_"+name)
        self.branch7x7dbl_2 = ConvBNLayer(num_channels=channels_7x7,  
                                       num_filters=channels_7x7, 
                                       filter_size=(7, 1), 
                                       padding = (3, 0),
                                       act="relu",
                                       name="inception_c_branch7x7dbl_2_"+name)
        self.branch7x7dbl_3 = ConvBNLayer(num_channels=channels_7x7, 
                                       num_filters=channels_7x7, 
                                       filter_size=(1, 7), 
                                       padding = (0, 3),
                                       act="relu",
                                       name="inception_c_branch7x7dbl_3_"+name)
        self.branch7x7dbl_4 = ConvBNLayer(num_channels=channels_7x7,  
                                       num_filters=channels_7x7, 
                                       filter_size=(7, 1), 
                                       padding = (3, 0),
                                       act="relu",
                                       name="inception_c_branch7x7dbl_4_"+name)
        self.branch7x7dbl_5 = ConvBNLayer(num_channels=channels_7x7, 
                                       num_filters=192, 
                                       filter_size=(1, 7), 
                                       padding = (0, 3),
                                       act="relu",
                                       name="inception_c_branch7x7dbl_5_"+name)
       
        self.branch_pool = AvgPool2D(kernel_size=3, stride=1, padding=1, exclusive=False)
        self.branch_pool_conv = ConvBNLayer(num_channels=num_channels,
                                       num_filters=192, 
                                       filter_size=1, 
                                       act="relu",
                                       name="inception_c_branch_pool_"+name)
        
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch1x1 = self.gsm(branch1x1)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = self.branch_pool(x)
        branch_pool = self.branch_pool_conv(branch_pool)

        outputs = paddle.concat([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=1)
        
        return outputs
    
class InceptionD(nn.Layer):
    def __init__(self, num_channels, name=None):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = ConvBNLayer(num_channels=num_channels, 
                                       num_filters=192, 
                                       filter_size=1, 
                                       act="relu",
                                       name="inception_d_branch3x3_1_"+name)
        self.gsm = gsmModule(fPlane=192, num_segments=16)

        self.branch3x3_2 = ConvBNLayer(num_channels=192, 
                                       num_filters=320, 
                                       filter_size=3, 
                                       stride=2,
                                       act="relu",
                                       name="inception_d_branch3x3_2_"+name)
        self.branch7x7x3_1 = ConvBNLayer(num_channels=num_channels, 
                                       num_filters=192, 
                                       filter_size=1, 
                                       act="relu",
                                       name="inception_d_branch7x7x3_1_"+name)
        self.branch7x7x3_2 = ConvBNLayer(num_channels=192,
                                       num_filters=192, 
                                       filter_size=(1, 7), 
                                       padding=(0, 3),
                                       act="relu",
                                       name="inception_d_branch7x7x3_2_"+name)
        self.branch7x7x3_3 = ConvBNLayer(num_channels=192, 
                                       num_filters=192, 
                                       filter_size=(7, 1), 
                                       padding=(3, 0),
                                       act="relu",
                                       name="inception_d_branch7x7x3_3_"+name)
        self.branch7x7x3_4 = ConvBNLayer(num_channels=192,  
                                       num_filters=192, 
                                       filter_size=3, 
                                       stride=2,
                                       act="relu",
                                       name="inception_d_branch7x7x3_4_"+name)
        self.branch_pool = MaxPool2D(kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.gsm(branch3x3)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = self.branch_pool(x)
        
        outputs = paddle.concat([branch3x3, branch7x7x3, branch_pool], axis=1)
        return outputs
    
class InceptionE(nn.Layer):
    def __init__(self, num_channels, name=None):
        super(InceptionE, self).__init__()
        self.branch1x1 = ConvBNLayer(num_channels=num_channels,
                                       num_filters=320, 
                                       filter_size=1, 
                                       act="relu",
                                       name="inception_e_branch1x1_"+name)
        self.gsm = gsmModule(fPlane=320, num_segments=16)

        self.branch3x3_1 = ConvBNLayer(num_channels=num_channels,
                                       num_filters=384, 
                                       filter_size=1, 
                                       act="relu",
                                       name="inception_e_branch3x3_1_"+name)
        self.branch3x3_2a = ConvBNLayer(num_channels=384, 
                                       num_filters=384, 
                                       filter_size=(1, 3), 
                                       padding=(0, 1),
                                       act="relu",
                                       name="inception_e_branch3x3_2a_"+name)
        self.branch3x3_2b = ConvBNLayer(num_channels=384, 
                                       num_filters=384, 
                                       filter_size=(3, 1), 
                                       padding=(1, 0),
                                       act="relu",
                                       name="inception_e_branch3x3_2b_"+name)
        
        self.branch3x3dbl_1 = ConvBNLayer(num_channels=num_channels, 
                                       num_filters=448, 
                                       filter_size=1, 
                                       act="relu",
                                       name="inception_e_branch3x3dbl_1_"+name)
        self.branch3x3dbl_2 = ConvBNLayer(num_channels=448, 
                                       num_filters=384, 
                                       filter_size=3, 
                                       padding=1,
                                       act="relu",
                                       name="inception_e_branch3x3dbl_2_"+name)
        self.branch3x3dbl_3a = ConvBNLayer(num_channels=384,
                                       num_filters=384, 
                                       filter_size=(1, 3), 
                                       padding=(0, 1),
                                       act="relu",
                                       name="inception_e_branch3x3dbl_3a_"+name)
        self.branch3x3dbl_3b = ConvBNLayer(num_channels=384,
                                       num_filters=384, 
                                       filter_size=(3, 1), 
                                       padding=(1, 0),
                                       act="relu",
                                       name="inception_e_branch3x3dbl_3b_"+name)
        self.branch_pool = AvgPool2D(kernel_size=3, stride=1, padding=1, exclusive=False)
        self.branch_pool_conv = ConvBNLayer(num_channels=num_channels, 
                                       num_filters=192, 
                                       filter_size=1, 
                                       act="relu",
                                       name="inception_e_branch_pool_"+name)
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch1x1 = self.gsm(branch1x1)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = paddle.concat(branch3x3, axis=1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = paddle.concat(branch3x3dbl, axis=1)

        branch_pool = self.branch_pool(x)
        branch_pool = self.branch_pool_conv(branch_pool)

        outputs = paddle.concat([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=1)
        return outputs   

    
class InceptionV3(nn.Layer):
    def __init__(self, class_dim=48, seg_num=16, seglen=1, modality="RGB"):
        super(InceptionV3, self).__init__()

        self.seg_num = seg_num

        self.seglen = seglen
        self.modality = modality
        self.channels = 3 * self.seglen if self.modality == "RGB" else 2 * self.seglen

        self.inception_a_list = [[192, 256, 288], [32, 64, 64]]
        self.inception_c_list = [[768, 768, 768, 768], [128, 160, 160, 192]]
        
        self.inception_stem = InceptionStem(self.channels)
        self.inception_block_list = []
        for i in range(len(self.inception_a_list[0])):
            inception_a = self.add_sublayer("inception_a_"+str(i+1),
                                           InceptionA(self.inception_a_list[0][i], 
                                                      self.inception_a_list[1][i], 
                                                      name=str(i+1)))
            self.inception_block_list.append(inception_a)
        inception_b = self.add_sublayer("nception_b_1",
                                        InceptionB(288, name="1"))
        self.inception_block_list.append(inception_b)
        
        for i in range(len(self.inception_c_list[0])):
            inception_c = self.add_sublayer("inception_c_"+str(i+1),
                                           InceptionC(self.inception_c_list[0][i], 
                                                      self.inception_c_list[1][i], 
                                                      name=str(i+1)))
            self.inception_block_list.append(inception_c)
        inception_d = self.add_sublayer("inception_d_1",
                                        InceptionD(768, name="1"))
        self.inception_block_list.append(inception_d)
        inception_e = self.add_sublayer("inception_e_1", 
                                        InceptionE(1280, name="1"))
        self.inception_block_list.append(inception_e)
        inception_e = self.add_sublayer("inception_e_2",
                                        InceptionE(2048, name="2"))
        self.inception_block_list.append(inception_e)
 
        self.gap = AdaptiveAvgPool2D(1)
        self.drop = Dropout(p=0.2, mode="downscale_in_infer")
        # self.drop = Dropout(p=0.7, mode="downscale_in_infer")
        stdv = 1.0 / math.sqrt(2048 * 1.0)
        self.out = Linear(
            2048,
            class_dim,
            weight_attr=ParamAttr(
                initializer=Uniform(-stdv, stdv), name="fc_weights"),
            bias_attr=ParamAttr(name="fc_offset"))

    def forward(self, x, label=None):
        
        x = paddle.reshape(x, [-1, x.shape[2]*self.seglen, x.shape[3], x.shape[4]])

        y = self.inception_stem(x)
        for inception_block in self.inception_block_list:
           y = inception_block(y)
        y = self.gap(y)
        # y = paddle.reshape(y, shape=[-1, 2048])
        # print(y.shape)
        y = paddle.reshape(y, [-1, self.seg_num//self.seglen, y.shape[1]])

        # print(y.shape)

        y = paddle.mean(y, axis=1)

        # print(y.shape)
        y = self.drop(y)
        y = self.out(y)

        y = paddle.nn.functional.softmax(y)
        # print('y.shape', y.shape)
        # print('label.shape', label.shape)
        
        if label is not None:
            acc = paddle.metric.accuracy(input=y, label=label)
            return y, acc
        else:
            return y
        return y

if __name__ == '__main__':
    network = InceptionV3(seglen=1)
    img = paddle.zeros([16, 16, 3, 299, 299])
    outs = network(img)
    print(outs.shape)