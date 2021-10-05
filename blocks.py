from utils.tools import canny_edge_detector
import torch

class EncoderDenseBlock(torch.nn.Module):
  def __init__(self, input_channels, output_channels, out_fn=True):
    super(EncoderDenseBlock, self).__init__()
    self.batch_norm1 = torch.nn.BatchNorm2d(output_channels)
    self.conv_1 = torch.nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
    if out_fn:
      self.conv_2 = torch.nn.Conv2d(input_channels+output_channels, output_channels, kernel_size=3, stride=1, padding=1)
    else:
      self.conv_2 = torch.nn.utils.spectral_norm(torch.nn.Conv2d(input_channels+output_channels, output_channels, kernel_size=3, stride=1, padding=1))
    self.batch_norm2 = torch.nn.BatchNorm2d(output_channels)
    self.activation_function = torch.nn.LeakyReLU()
    self.flag = out_fn

  def forward(self, x):
    conv1 = self.activation_function(self.batch_norm1(self.conv_1(x)))
    block = self.conv_2(torch.cat([x, conv1], dim=1))
    ret = self.activation_function(self.batch_norm2(block)) if self.flag else block
    return ret

class EncoderTransitionBlock(torch.nn.Module):
  def __init__(self, input_channels, output_channels):
    super(EncoderTransitionBlock, self).__init__()
    self.batch_norm = torch.nn.BatchNorm2d(input_channels)
    self.conv = torch.nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1)
    self.pooling_layer = torch.nn.AvgPool2d(2)
    self.activation_function = torch.nn.LeakyReLU()

  def forward(self, x):
    x = self.activation_function(self.batch_norm(self.conv(x)))
    return self.pooling_layer(x)

class EncoderTransitionAttentionBlock(torch.nn.Module):
  def __init__(self, input_channels, output_channels):
    super(EncoderTransitionAttentionBlock, self).__init__()
    self.conv_gate = torch.nn.utils.parametrizations.spectral_norm(torch.nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1))
    self.conv_feature = torch.nn.utils.parametrizations.spectral_norm(torch.nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1))
    self.pooling_layer = torch.nn.AvgPool2d(2)
    self.activation_function = torch.nn.LeakyReLU()
    self.sigmoid = torch.nn.Sigmoid()

  def forward(self, x):
    x = self.pooling_layer(x)
    gating = self.sigmoid(self.conv_gate(x))
    feature = self.activation_function(self.conv_feature(x))
    return torch.multiply(feature, gating)

class EncoderBlock(torch.nn.Module):
  def __init__(self, input_channels, output_channels):
    super(EncoderBlock, self).__init__()
    self.conv_1 = torch.nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
    self.conv_2 = torch.nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)
    self.batch_norm = torch.nn.BatchNorm2d(output_channels)
    self.pooling_layer = torch.nn.AvgPool2d(2)
    self.relu_1 = torch.nn.LeakyReLU()
    self.relu_2 = torch.nn.LeakyReLU()

  def forward(self, x):
    x = self.relu_1(self.conv_1(self.pooling_layer(x)))
    x = self.relu_2(self.batch_norm(self.conv_2(x)))
    return x

class AttentionBlock(torch.nn.Module):
  def __init__(self, input_channels, output_channels):
    super(AttentionBlock, self).__init__()
    self.conv_q = torch.nn.utils.spectral_norm(torch.nn.Conv1d(input_channels, output_channels, kernel_size=1, stride=1, bias=False))
    self.conv_v = torch.nn.utils.spectral_norm(torch.nn.Conv1d(input_channels, input_channels, kernel_size=1, stride=1, bias=False))
    self.conv_k = torch.nn.utils.spectral_norm(torch.nn.Conv1d(input_channels, output_channels, kernel_size=1, stride=1, bias=False))
    self.softmax = torch.nn.Softmax(dim=1)
    self.gamma = torch.nn.Parameter(torch.tensor([0.0]))

  def forward(self, x):
    size = x.shape
    x = x.view(*size[:2],-1)

    f = self.conv_q(x)
    g = self.conv_k(x)
    h = self.conv_v(x)

    print(f.shape)
    print(g.shape)
    beta = self.softmax(torch.bmm(torch.transpose(f, 1, 2), g))
    o = self.gamma*torch.bmm(h, beta) + x
    return o.view(*size).contiguous()

class EncoderAttBlock(torch.nn.Module):
  def __init__(self, input_channels, output_channels):
    super(EncoderAttBlock, self).__init__()
    self.conv_gate = torch.nn.utils.parametrizations.spectral_norm(torch.nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1))
    self.conv_feature = torch.nn.utils.parametrizations.spectral_norm(torch.nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1))

    self.pooling_layer = torch.nn.AvgPool2d(2)
    self.relu = torch.nn.LeakyReLU()
    # self.softmax = torch.nn.Softmax(dim=1)
    self.sigmoid = torch.nn.Sigmoid()

  def forward(self, x):
    x = self.pooling_layer(x)
    gating = self.sigmoid(self.conv_gate(x))
    feature = self.relu(self.conv_feature(x))
    output = torch.multiply(feature, gating)
    return output

class DecoderAttBlock(torch.nn.Module):
  def __init__(self, input_channels, output_channels):
    super(DecoderAttBlock, self).__init__()
    self.conv_1 = torch.nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1)
    self.conv_2 = torch.nn.Conv2d(2*output_channels, 2*output_channels, kernel_size=3, stride=1, padding=1)
    self.conv_3 = torch.nn.Conv2d(2*output_channels, output_channels, kernel_size=1, stride=1)
    self.batch_norm = torch.nn.BatchNorm2d(2*output_channels)
    self.unpooling_layer = torch.nn.MaxUnpool2d(2)
    self.relu_1 = torch.nn.LeakyReLU()
    self.relu_2 = torch.nn.LeakyReLU()
    self.relu_3 = torch.nn.LeakyReLU()

    # self.softmax = torch.nn.Softmax(dim=1)
    self.sigmoid = torch.nn.Sigmoid()
    self.conv_gate = torch.nn.utils.parametrizations.spectral_norm(torch.nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1))

  def forward(self, x, skipped_input):
    x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

    x = self.relu_1(self.conv_1(x))

    x = torch.cat([x, skipped_input], dim=1)

    x = self.relu_2(self.batch_norm(self.conv_2(x))) 
    gating = self.sigmoid(self.conv_gate(x))   
    x = self.relu_3(self.conv_3(x)) 

    output = torch.multiply(gating, x)
    return output

class NewDecoderBlock(torch.nn.Module):
  def __init__(self, input_channels):
    super(NewDecoderBlock, self).__init__()
    self.att = AttentionBlock(input_channels//2, input_channels//16)
    self.batch_norm = torch.nn.BatchNorm2d(input_channels)
    self.relu = torch.nn.LeakyReLU()
    self.conv1 = torch.nn.Conv2d(input_channels, input_channels//2, kernel_size=1, stride=1)

  def forward(self, x, skipped_input):
    # Avg unpool x
    x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

    # 1x1 convolution to reduce the number of channels
    x = self.relu(self.conv1(self.batch_norm(x)))

    # Concatenate x and the skipped connection
    x = x + skipped_input

    # Apply the attention layer
    x = self.att(x)
    return x

class DecoderBlock(torch.nn.Module):
  def __init__(self, input_channels, output_channels):
    super(DecoderBlock, self).__init__()
    self.conv_1 = torch.nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1)
    self.batch_norm1 = torch.nn.BatchNorm2d(output_channels)
    self.conv_2 = torch.nn.Conv2d(2*output_channels, 2*output_channels, kernel_size=3, stride=1, padding=1)
    self.batch_norm2 = torch.nn.BatchNorm2d(2*output_channels)
    self.conv_3 = torch.nn.Conv2d(2*output_channels, output_channels, kernel_size=1, stride=1)
    self.batch_norm3 = torch.nn.BatchNorm2d(output_channels)
    self.relu_1 = torch.nn.LeakyReLU()
    self.relu_2 = torch.nn.LeakyReLU()
    self.relu_3 = torch.nn.LeakyReLU()

  def forward(self, x, skipped_input):
    x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
    x = self.relu_1(self.batch_norm1(self.conv_1(x)))

    x = torch.cat([x, skipped_input], dim=1)
    x = self.relu_2(self.batch_norm2(self.conv_2(x)))    
    x = self.relu_3(self.batch_norm3(self.conv_3(x))) 
    return x

class WeightMapPredictor(torch.nn.Module):
  def __init__(self, channels=32):
    super(WeightMapPredictor, self).__init__()
    self.convd_1 = torch.nn.Conv2d(3, channels, kernel_size=3, dilation=2, padding=2)
    self.convd_2 = torch.nn.Conv2d(channels, channels, kernel_size=3, dilation=2, padding=2)
    self.convd_3 = torch.nn.Conv2d(channels, channels, kernel_size=3, dilation=2, padding=2)

    self.conv_1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    self.conv_2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    self.conv_3 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    self.activation = torch.nn.ReLU()
    self.conv_out = torch.nn.Conv2d(channels, 4, kernel_size=1)

  def forward(self, input):
    edges = canny_edge_detector(input)
    x = self.activation(self.convd_1(input))
    # print(x.shape)
    x = self.activation(self.convd_2(x))
    # print(x.shape)
    x = self.activation(self.convd_3(x))
    # print(x.shape)

    x = self.activation(self.conv_1(x))
    # print(x.shape)
    x = self.activation(self.conv_2(x))
    # print(x.shape)
    x = self.activation(self.conv_3(x))
    # print(x.shape)

    res_map = self.conv_out(x)
    # print(res_map.shape)

    # concatenate the edges by adding them to each dimension of the residual maps
    edges = torch.unsqueeze(edges, dim=1)
    weight_maps = torch.clamp(res_map+edges, min=0.0, max=1.0)
    del edges
    del res_map
    return weight_maps

class RecursivePropagator(torch.nn.Module):
  def __init__(self, repetitions=3):
    super(RecursivePropagator, self).__init__()
    self.repetitions = repetitions

  def forward(self, input, weight_maps):
    channels = weight_maps.shape[1]
    assert channels == 4

    signal = input
    w_lr, w_rl, w_tb, w_bt = torch.split(weight_maps, 1, dim=1)
    for _ in range(self.repetitions):
      # Pad the left side and apply the propagation from left to right
      aux = torch.nn.functional.pad(signal, (1, 0), "constant", 0)
      signal_shifted = aux[:, :, :, :-1]
      signal = w_lr*signal+(1-w_lr)*signal_shifted

      # Pad the right side and apply the propagation from right to left
      aux = torch.nn.functional.pad(signal, (0, 1), "constant", 0)
      signal_shifted = aux[:, :, :, 1:]
      signal = w_rl*signal+(1-w_rl)*signal_shifted

      # Pad the top side and apply the propagation from top to bottom
      aux = torch.nn.functional.pad(signal, (0, 0, 1, 0), "constant", 0)
      signal_shifted = aux[:, :, :-1, :]
      signal = w_tb*signal+(1-w_tb)*signal_shifted

      # Pad the bottom side and apply the propagation from bottom to top
      aux = torch.nn.functional.pad(signal, (0, 0, 0, 1), "constant", 0)
      signal_shifted = aux[:, :, 1:, :]
      signal = w_bt*signal+(1-w_bt)*signal_shifted

    del aux
    del signal_shifted
    return signal
