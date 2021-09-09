from blocks import *

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
