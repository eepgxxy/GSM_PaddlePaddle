import paddle
from paddle import nn

class gsmModule(nn.Layer):
    def __init__(self, fPlane, num_segments=3):
        super(gsmModule, self).__init__()

        self.conv3D = nn.Conv3D(fPlane, 2, (3, 3, 3), stride=1,
                                padding=(1, 1, 1), groups=2, weight_attr=paddle.nn.initializer.Constant(value=0.0))
        self.tanh = nn.Tanh()
        self.fPlane = fPlane
        self.num_segments = num_segments
        self.bn = nn.BatchNorm3D(num_features=fPlane)
        self.relu = nn.ReLU()

    def lshift_zeroPad(self, x):
        x_tensor = paddle.zeros(shape=[x.shape[0], x.shape[1], 1, x.shape[3], x.shape[4]]) 
        return paddle.concat([x[:,:,1:], x_tensor], axis=2)

    def rshift_zeroPad(self, x):
        x_tensor = paddle.zeros(shape=[x.shape[0], x.shape[1], 1, x.shape[3], x.shape[4]]) 
        return paddle.concat([x_tensor, x[:,:,:-1]], axis=2)

    def forward(self, x):
        batchSize = x.shape[0] // self.num_segments
        shape = x.shape[1], x.shape[2], x.shape[3]
        assert  shape[0] == self.fPlane
        x = paddle.reshape(x=x, shape=[batchSize, self.num_segments, *shape])

        x = paddle.reshape(x=x, shape=[x.shape[0], x.shape[2], x.shape[1], x.shape[3], x.shape[4]])
        x_bn = self.bn(x)
        x_bn_relu = self.relu(x_bn)
        gate = self.tanh(self.conv3D(x_bn_relu))

        gate_group1 = gate[:, 0].unsqueeze(1)
        gate_group2 = gate[:, 1].unsqueeze(1)

        x_group1 = x[:, :self.fPlane // 2]
        x_group2 = x[:, self.fPlane // 2:]

        y_group1 = gate_group1 * x_group1
        y_group2 = gate_group2 * x_group2

        r_group1 = x_group1 - y_group1
        r_group2 = x_group2 - y_group2

        y_group1 = self.lshift_zeroPad(y_group1) + r_group1
        y_group2 = self.rshift_zeroPad(y_group2) + r_group2

        y_group1 = paddle.reshape(x=y_group1, shape=[batchSize, self.fPlane//2, self.num_segments, *shape[1:]])

        y_group2 = paddle.reshape(x=y_group2, shape=[batchSize, self.fPlane//2, self.num_segments, *shape[1:]])

        y = paddle.concat(x=[y_group1, y_group2], axis=1)

        y = paddle.reshape(x=y, shape=[y.shape[0],y.shape[2],y.shape[1],y.shape[3],y.shape[4]])

        y = paddle.reshape(x=y, shape=[batchSize*self.num_segments, *shape])

        return y