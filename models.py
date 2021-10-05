from blocks import *

class DenseNN(torch.nn.Module):
  def __init__(self, repetitions=7):
    super(DenseNN, self).__init__()

    # # Initial convolution and pooling
    # self.initial_conv = torch.nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
    # self.batch_norm = torch.nn.BatchNorm2d(4)
    # self.activation_function = torch.nn.LeakyReLU()
    # self.initial_pooling = torch.nn.AvgPool2d(2)
    self.enc0_dense1 = EncoderDenseBlock(4, 16)
    self.enc0_dense2 = EncoderDenseBlock(16, 16)
    self.enc0_dense3 = EncoderDenseBlock(16, 16)
    self.enc0_trans = EncoderTransitionBlock(16, 8)

    #Input: 272x192x8
    self.enc1_dense1 = EncoderDenseBlock(8, 32)
    self.enc1_dense2 = EncoderDenseBlock(32, 32)
    self.enc1_dense3 = EncoderDenseBlock(32, 32)
    self.enc1_trans = EncoderTransitionBlock(32, 16)

    #Input: 136x96x16
    self.enc2_dense1 = EncoderDenseBlock(16, 64)
    self.enc2_dense2 = EncoderDenseBlock(64, 64)
    # self.enc2_dense3 = EncoderDenseBlock(64, 64)
    self.enc2_trans = EncoderTransitionBlock(64, 32)

    #Input: 68x48x32
    self.enc3_dense1 = EncoderDenseBlock(32, 128)
    self.enc3_dense2 = EncoderDenseBlock(128, 128)
    # self.enc3_dense3 = EncoderDenseBlock(128, 128)
    self.enc3_trans = EncoderTransitionBlock(128, 64)

    #Input: 34x24x64
    self.enc4_dense1 = EncoderDenseBlock(64, 256)
    self.enc4_dense2 = EncoderDenseBlock(256, 256)
    # self.enc4_dense3 = EncoderDenseBlock(256, 256)
    self.enc4_trans = EncoderTransitionBlock(256, 128)

    #17x12x128
    self.dec1 = DecoderBlock(128, 64)

    #34x24x64
    self.dec2 = DecoderBlock(64, 32)

    #68x48x32
    self.dec3 = DecoderBlock(32, 16)

    #136x96x16
    self.dec4 = DecoderBlock(16, 8)

    # self.dec5 = DecoderBlock(8, 1)

    #272x192x8
    self.conv1 = torch.nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
    self.batch_norm1 = torch.nn.BatchNorm2d(8)
    self.conv2 = torch.nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1)
    self.batch_norm2 = torch.nn.BatchNorm2d(12)

    #544x384x9
    self.outputLayer = torch.nn.Conv2d(12, 1, kernel_size=1, stride=1)
    self.batch_norm3 = torch.nn.BatchNorm2d(1)

    self.relu_1 = torch.nn.LeakyReLU()
    self.relu_2 = torch.nn.LeakyReLU()
    self.relu_3 = torch.nn.LeakyReLU()

    # Edge refinement
    self.w_pred = WeightMapPredictor(channels=32)
    self.propagator = RecursivePropagator(repetitions=repetitions)

  def forward(self, x):
    # input = self.initial_pooling(self.activation_function(self.initial_conv(self.batch_norm(x))))
    input = self.enc0_trans(self.enc0_dense3(self.enc0_dense2(self.enc0_dense1(x))))
    enc_step_1 = self.enc1_trans(self.enc1_dense3(self.enc1_dense2(self.enc1_dense1(input))))
    # enc_step_1 = self.enc1_trans(self.enc1_dense2(self.enc1_dense1(input)))

    # enc_step_2 = self.enc2_trans(self.enc2_dense3(self.enc2_dense2(self.enc2_dense1(enc_step_1))))
    enc_step_2 = self.enc2_trans(self.enc2_dense2(self.enc2_dense1(enc_step_1)))

    # enc_step_3 = self.enc3_trans(self.enc3_dense3(self.enc3_dense2(self.enc3_dense1(enc_step_2))))
    enc_step_3 = self.enc3_trans(self.enc3_dense2(self.enc3_dense1(enc_step_2)))

    # enc_step_4 = self.enc4_trans(self.enc4_dense3(self.enc4_dense2(self.enc4_dense1(enc_step_3))))
    enc_step_4 = self.enc4_trans(self.enc4_dense2(self.enc4_dense1(enc_step_3)))

    dec_step_1 = self.dec1(enc_step_4, enc_step_3)
    del enc_step_4
    del enc_step_3
    dec_step_2 = self.dec2(dec_step_1, enc_step_2)
    del enc_step_2
    del dec_step_1
    dec_step_3 = self.dec3(dec_step_2, enc_step_1)
    del enc_step_1
    del dec_step_2
    dec_step_4 = self.dec4(dec_step_3, input)
    del input
    del dec_step_3

    y = torch.nn.functional.interpolate(dec_step_4, scale_factor=2, mode='bilinear', align_corners=True)
    del dec_step_4
    y = self.relu_1(self.batch_norm1(self.conv1(y)))
    y = torch.cat([y, x], dim=1)
    y = self.relu_2(self.batch_norm2(self.conv2(y)))
    
    depth = self.relu_3(self.batch_norm3(self.outputLayer(y)))
    del y

    # depth = self.dec5(dec_step_4, x)

    rgb = x[:, 0:3, :, :]
    w_maps = self.w_pred(rgb)
    final_depth = self.propagator(depth, w_maps)

    del w_maps

    return final_depth, depth

class DenseAttNN(torch.nn.Module):
  def __init__(self):
    super(DenseAttNN, self).__init__()

    # Initial convolution and pooling
    self.initial_conv = torch.nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
    self.batch_norm = torch.nn.BatchNorm2d(4)
    self.activation_function = torch.nn.LeakyReLU()
    self.initial_pooling = torch.nn.AvgPool2d(2)

    #Input: 272x192x8
    self.enc1_dense1_f = EncoderDenseBlock(8, 32)
    self.enc1_dense2_f = EncoderDenseBlock(32, 32)
    self.enc1_dense3_f = EncoderDenseBlock(32, 32, False)
    self.enc1_dense3_g = EncoderDenseBlock(32, 32, False)
    self.enc1_trans = EncoderTransitionBlock(32, 16)

    #Input: 136x96x16
    self.enc2_dense1_f = EncoderDenseBlock(16, 64)
    self.enc2_dense2_f = EncoderDenseBlock(64, 64, False)
    self.enc2_dense2_g = EncoderDenseBlock(64, 64, False)
    # self.enc2_dense3 = EncoderDenseBlock(64, 64)
    self.enc2_trans = EncoderTransitionBlock(64, 32)

    #Input: 68x48x32
    self.enc3_dense1_f = EncoderDenseBlock(32, 128)
    self.enc3_dense2_f = EncoderDenseBlock(128, 128, False)
    self.enc3_dense2_g = EncoderDenseBlock(128, 128, False)

    # self.enc3_dense3 = EncoderDenseBlock(128, 128)
    self.enc3_trans = EncoderTransitionBlock(128, 64)

    #Input: 34x24x64
    self.enc4_dense1_f = EncoderDenseBlock(64, 256)
    self.enc4_dense2_f = EncoderDenseBlock(256, 256, False)
    self.enc4_dense2_g = EncoderDenseBlock(256, 256, False)
    # self.enc4_dense3 = EncoderDenseBlock(256, 256)
    self.enc4_trans = EncoderTransitionBlock(256, 128)

    #17x12x128
    self.dec1 = DecoderBlock(128, 64)

    #34x24x64
    self.dec2 = DecoderBlock(64, 32)

    #68x48x32
    self.dec3 = DecoderBlock(32, 16)

    #136x96x16
    self.dec4 = DecoderBlock(16, 8)

    #272x192x8
    self.conv1 = torch.nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
    self.conv2 = torch.nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1)

    #544x384x9
    self.outputLayer = torch.nn.Conv2d(12, 1, kernel_size=1, stride=1)

    self.relu = torch.nn.LeakyReLU()
    self.sigmoid = torch.nn.Sigmoid()

    # Edge refinement
    self.w_pred = WeightMapPredictor(channels=32)
    self.propagator = RecursivePropagator()

  def forward(self, x):
    input = self.initial_pooling(self.activation_function(self.initial_conv(self.batch_norm(x))))
    enc_step_1 = self.enc1_dense2_f(self.enc1_dense1_f(input))
    enc_step_1_f = self.relu(self.enc1_dense3_f(enc_step_1))
    enc_step_1_g = self.sigmoid(self.enc1_dense3_g(enc_step_1))
    enc_step_1 = self.enc1_trans(torch.mul(enc_step_1_f, enc_step_1_g))

    enc_step_2 = self.enc2_dense1_f(enc_step_1)
    enc_step_2_f = self.relu(self.enc2_dense2_f(enc_step_2))
    enc_step_2_g = self.sigmoid(self.enc2_dense2_g(enc_step_2))
    enc_step_2 = self.enc2_trans(torch.mul(enc_step_2_f, enc_step_2_g))

    enc_step_3 = self.enc3_dense1_f(enc_step_2)
    enc_step_3_f = self.relu(self.enc3_dense2_f(enc_step_3))
    enc_step_3_g = self.sigmoid(self.enc3_dense2_g(enc_step_3))
    enc_step_3 = self.enc3_trans(torch.mul(enc_step_3_f, enc_step_3_g))

    enc_step_4 = self.enc4_dense1_f(enc_step_3)
    enc_step_4_f = self.relu(self.enc4_dense2_f(enc_step_4))
    enc_step_4_g = self.sigmoid(self.enc4_dense2_g(enc_step_4))
    enc_step_4 = self.enc4_trans(torch.mul(enc_step_4_f, enc_step_4_g))

    dec_step_1 = self.dec1(enc_step_4, enc_step_3)
    del enc_step_4
    del enc_step_3
    dec_step_2 = self.dec2(dec_step_1, enc_step_2)
    del enc_step_2
    del dec_step_1
    dec_step_3 = self.dec3(dec_step_2, enc_step_1)
    del enc_step_1
    del dec_step_2
    dec_step_4 = self.dec4(dec_step_3, input)
    del input
    del dec_step_3

    y = torch.nn.functional.interpolate(dec_step_4, scale_factor=2, mode='bilinear', align_corners=True)
    del dec_step_4
    y = self.relu(self.conv1(y))
    y = torch.cat([y, x], dim=1)
    y = self.relu(self.conv2(y))
    
    depth = self.relu(self.outputLayer(y))
    del y

    rgb = x[:, 0:3, :, :]
    w_maps = self.w_pred(rgb)
    final_depth = self.propagator(depth, w_maps)
    del depth
    del w_maps

    return final_depth  


class DepthCompletionNN(torch.nn.Module):
  def __init__(self):
    super(DepthCompletionNN, self).__init__()

    #544x384x4
    self.enc1 = EncoderBlock(4, 8)

    #272x192x8
    self.enc2 = EncoderBlock(8, 16)

    #136x96x16
    self.enc3 = EncoderBlock(16, 32)

    #68x48x32
    self.enc4 = EncoderBlock(32, 64)

    #34x24x64
    self.enc5 = EncoderBlock(64, 128)

    #17x12x128
    self.dec1 = DecoderBlock(128, 64)

    #34x24x64
    self.dec2 = DecoderBlock(64, 32)

    #68x48x32
    self.dec3 = DecoderBlock(32, 16)

    #136x96x16
    self.dec4 = DecoderBlock(16, 8)

    #272x192x8
    self.conv1 = torch.nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
    self.conv2 = torch.nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1)

    #544x384x9
    self.outputLayer = torch.nn.Conv2d(12, 1, kernel_size=1, stride=1)

    self.relu_1 = torch.nn.LeakyReLU()
    self.relu_2 = torch.nn.LeakyReLU()
    self.relu_3 = torch.nn.ReLU()

  def forward(self, x):
    enc_step_1 = self.enc1(x)
    enc_step_2 = self.enc2(enc_step_1)
    enc_step_3 = self.enc3(enc_step_2)
    enc_step_4 = self.enc4(enc_step_3)
    enc_step_5 = self.enc5(enc_step_4)

    dec_step_1 = self.dec1(enc_step_5, enc_step_4)
    dec_step_2 = self.dec2(dec_step_1, enc_step_3)
    dec_step_3 = self.dec3(dec_step_2, enc_step_2)
    dec_step_4 = self.dec4(dec_step_3, enc_step_1)

    y = torch.nn.functional.interpolate(dec_step_4, scale_factor=2, mode='bilinear', align_corners=True)
    y = self.relu_1(self.conv1(y))
    y = torch.cat([y, x], dim=1)
    y = self.relu_2(self.conv2(y))

    depth = self.relu_3(self.outputLayer(y))

    return depth

class DepthEstimationNN(torch.nn.Module):
  def __init__(self):
    super(DepthEstimationNN, self).__init__()

    #544x384x4
    self.enc1 = EncoderAttBlock(3, 8)

    #272x192x8
    self.enc2 = EncoderAttBlock(8, 16)

    #136x96x16
    self.enc3 = EncoderAttBlock(16, 32)

    #68x48x32
    self.enc4 = EncoderAttBlock(32, 64)

    #34x24x64
    self.enc5 = EncoderAttBlock(64, 128)

    #17x12x128
    self.dec1 = DecoderAttBlock(128, 64)

    #34x24x64
    self.dec2 = DecoderAttBlock(64, 32)

    #68x48x32
    self.dec3 = DecoderAttBlock(32, 16)

    #136x96x16
    self.dec4 = DecoderAttBlock(16, 8)

    #272x192x8
    self.conv1 = torch.nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
    self.conv2 = torch.nn.Conv2d(11, 11, kernel_size=3, stride=1, padding=1)

    #544x384x9
    self.outputLayer = torch.nn.Conv2d(11, 1, kernel_size=1, stride=1)

    self.relu_1 = torch.nn.LeakyReLU()
    self.relu_2 = torch.nn.LeakyReLU()
    self.relu_3 = torch.nn.ReLU()

  def forward(self, x):
    enc_step_1 = self.enc1(x)
    enc_step_2 = self.enc2(enc_step_1)
    enc_step_3 = self.enc3(enc_step_2)
    enc_step_4 = self.enc4(enc_step_3)
    enc_step_5 = self.enc5(enc_step_4)

    dec_step_1 = self.dec1(enc_step_5, enc_step_4)
    dec_step_2 = self.dec2(dec_step_1, enc_step_3)
    dec_step_3 = self.dec3(dec_step_2, enc_step_2)
    dec_step_4 = self.dec4(dec_step_3, enc_step_1)

    y = torch.nn.functional.interpolate(dec_step_4, scale_factor=2, mode='bilinear', align_corners=True)
    y = self.relu_1(self.conv1(y))
    y = torch.cat([y, x], dim=1)
    y = self.relu_2(self.conv2(y))
    
    depth = self.relu_3(self.outputLayer(y))

    return depth

class AttentionNN(torch.nn.Module):
  def __init__(self):
    super(AttentionNN, self).__init__()

    #544x384x4
    self.enc1 = EncoderAttBlock(4, 8)

    #272x192x8
    self.enc2 = EncoderAttBlock(8, 16)

    #136x96x16
    self.enc3 = EncoderAttBlock(16, 32)

    #68x48x32
    self.enc4 = EncoderAttBlock(32, 64)

    #34x24x64
    self.enc5 = EncoderAttBlock(64, 128)

    #17x12x128
    self.dec1 = DecoderAttBlock(128, 64)

    #34x24x64
    self.dec2 = DecoderAttBlock(64, 32)

    #68x48x32
    self.dec3 = DecoderAttBlock(32, 16)

    #136x96x16
    self.dec4 = DecoderAttBlock(16, 8)

    #272x192x8
    self.conv1 = torch.nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
    self.conv2 = torch.nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1)

    #544x384x9
    self.outputLayer = torch.nn.Conv2d(12, 1, kernel_size=1, stride=1)

    self.relu_1 = torch.nn.LeakyReLU()
    self.relu_2 = torch.nn.LeakyReLU()
    self.relu_3 = torch.nn.ReLU()

  def forward(self, x):
    enc_step_1 = self.enc1(x)
    enc_step_2 = self.enc2(enc_step_1)
    enc_step_3 = self.enc3(enc_step_2)
    enc_step_4 = self.enc4(enc_step_3)
    enc_step_5 = self.enc5(enc_step_4)

    dec_step_1 = self.dec1(enc_step_5, enc_step_4)
    dec_step_2 = self.dec2(dec_step_1, enc_step_3)
    dec_step_3 = self.dec3(dec_step_2, enc_step_2)
    dec_step_4 = self.dec4(dec_step_3, enc_step_1)

    y = torch.nn.functional.interpolate(dec_step_4, scale_factor=2, mode='bilinear', align_corners=True)
    y = self.relu_1(self.conv1(y))
    y = torch.cat([y, x], dim=1)
    y = self.relu_2(self.conv2(y))
    
    depth = self.relu_3(self.outputLayer(y))

    return depth
