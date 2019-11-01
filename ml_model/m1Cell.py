import torch
import torch.nn as nn

def get_same_padding(image_size, filter_size, stride):
    padding_x = (image_size[0] * stride - 1 - image_size[0] + filter_size) / 2
    padding_y = (image_size[1] * stride - 1 - image_size[1] + filter_size) / 2

    return [padding_x, padding_y]

class M1Cell(nn.Module):
    def __init__(self, image_size, kernels, output_channels, stride):
        super(M1Cell, self).__init__()

        # Unpack the kernels
        kernel_3, kernel_5, kernel_7 = kernels

        # padding for 3x3 filter
        padding_3 = get_same_padding(image_size, kernel_1, stride)
        # padding for 5x5 filter
        padding_5 = get_same_padding(image_size, kernel_2, stride)
        # padding for 7x7 filter
        padding_7 = get_same_padding(image_size, kernel_3, stride)
        # filters
        # A1 * x layer
        self.conv_1 = nn.Conv2d(output_channels, output_channels,
            kernel_size=kernel_3, padding=padding_3)
        # A2 * x layer
        self.conv_2 = nn.Conv2d(output_channels, output_channels,
            kernel_size=kernel_5, padding=padding_5)
        # A3 * x layer
        self.conv_3 = nn.Conv2d(output_channels, output_channels,
            kernel_size=kernel_7, padding=padding_7)

        # B1 * x² layer
        self.conv_xx_1 = nn.Conv2d(output_channels, output_channels,
            kernel_size=kernel_3, padding=padding_3)
        # B2 * x² layer
        self.conv_xx_2 = nn.Conv2d(output_channels, output_channels,
            kernel_size=kernel_5, padding=padding_5)
        # B3 * x² layer
        self.conv_xx_3 = nn.Conv2d(output_channels, output_channels,
            kernel_size=kernel_7, padding=padding_7)

        # gamma_1 * (A_1 * x + B_1 * x^2 + C_1 * u)
        self.conv_g_1 = nn.Conv2d(output_channels, output_channels,
            kernel_size=kernel_3, padding=padding_3)
        # gamma_2 * (A_2 * x + B_2 * x^2 + C_2 * u)
        self.conv_g_2 = nn.Conv2d(output_channels, output_channels,
            kernel_size=kernel_5, padding=padding_5)
        # gamma_3 * (A_3 * x + B_3 * x^2 + C_3 * u)
        self.conv_g_3 = nn.Conv2d(output_channels, output_channels,
            kernel_size=kernel_7, padding=padding_7)

        # alpha_1 * x
        self.conv_a_1 = nn.Conv2d(output_channels, output_channels,
            kernel_size=kernel_3, padding=padding_3)
        # gamma_2 * (A_2 * x + B_2 * x^2 + C_2 * u)
        self.conv_a_2 = nn.Conv2d(output_channels, output_channels,
            kernel_size=kernel_5, padding=padding_5)
        # gamma_3 * (A_3 * x + B_3 * x^2 + C_3 * u)
        self.conv_a_3 = nn.Conv2d(output_channels, output_channels,
            kernel_size=kernel_7, padding=padding_7)

    def call(self, inputs, state):
        xx = state * state
        xxx = xx * state
        #print(state.get_shape())
        # A1 * x
        conv1 = self.conv_1(state)
        # A2 * x
        conv2 = self.conv_2(state)
        # A3 * x
        conv3 = self.conv_3(state)

        # B1 * x²
        convxx1 = self.conv_xx_1(xx)
        # B2 * x²
        convxx2 = self.conv_xx_2(xx)
        # B3 * x²
        convxx3 = self.conv_xx_3(xx)

        # First function
        sum_1 = conv1 + convxx1 + inputs
        f1 = torch.tanh(sum_1)
        # Second function
        sum_2 = conv2 + convxx2 + inputs
        f2 = torch.tanh(sum_2)
        # Third function
        sum_3 = conv3 + convxx3 + inputs
        f3 = torch.tanh(sum_3)

        # Conv gamma_1
        convg1 = self.conv_g_1(f1)
        # Conv gamma_2
        convg2 = self.conv_g_2(f2)
        # Conv gamma_3
        convg3 = self.conv_g_3(f3)

        # Get the nonlinear function
        nonlinear = convg1 + convg2 + convg3

        # Get the non-activation convolutions
        # alpha_1 * x
        n_out_1 = self.conv_a_1(state)
        # alpha_2 * x²
        n_out_2 = self.conv_a_2(xx)
        # alpha_3 * x³
        n_out_3 = self.conv_a_3(xxx)

        # Compute the final state
        state = n_out_1 + n_out_2 + n_out_3 + nonlinear

        return state
