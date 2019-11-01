import torch
import torch.nn as nn

import m1Cell

def get_same_padding(image_size, filter_size, stride):
    padding_x = (image_size[0] * stride - 1 - image_size[0] + filter_size) / 2
    padding_y = (image_size[1] * stride - 1 - image_size[1] + filter_size) / 2

    return [padding_x, padding_y]

class M1Classifier(nn.Module):
    def __init__(self, kernels, output_channels, stride
    batch_size, image_size, num_rec, num_layers, out_nodes):
        super(M1Classifier, self).__init__()
        # Unpack the kernels
        kernel_3, kernel_5, kernel_7 = kernels
        # unpack output nodes
        out_node1, num_classes = out_nodes
        # pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # First input layer stack
        self.input_layer_1 = nn.Conv2d(output_channels, output_channels,
            kernel_size=kernel_3, padding=padding_3)
        # Second input layer stack
        self.input_layer_2 = nn.Conv2d(output_channels, output_channels,
            kernel_size=kernel_5, padding=padding_5)
        # Third input layer stack
        self.input_layer_3 = nn.Conv2d(output_channels, output_channels,
            kernel_size=kernel_7, padding=padding_7)

        # Cell input and cell objects
        self.input_layer_list = []
        self.cell_list = []
        self.num_layers = num_layers
        for i in range(self.num_layers):
            self.input_layer_list.append(nn.Conv2d(output_channels,
                output_channels, kernel_size=kernel_7, padding=padding_7))
            self.cell_list.append(m1Cell.M1Cell(image_size, kernels, output_channels,
                stride))
        # expand cell output to 16 classes
        self.expand_features = nn.Conv2d(output_channels, output_channels,
            kernel_size=kernel_3, padding=padding_7)
        # flat layer
        self.flat_layer = nn.Flatten()
        # Drop out
        self.drp =  nn.Dropout(0.6)
        # classification layer
        self.fc_1 = nn.Linear(in_features=channels*28*28,
            out_features=out_node1)
        #self.fc_2 = tf.keras.layers.Dense(out_node1)
        self.fc_2 = nn.Linear(in_features=out_node1,
            out_features=num_classes)

    def call(self, u):
        # Compute the input layer output
        conv_input_1 = self.input_layer_1(u)
        conv_input_2 = self.input_layer_2(u)
        conv_input_3 = self.input_layer_3(u)

        # concatenate the computed features
        conv_input = torch.cat([conv_input_1, conv_input_2, conv_input_3,
            axis=3])
        conv_input = self.maxpool(conv_input)
        #conv_input = self.maxpool_2d(conv_input)
        for i in range(self.num_layers):
            conv_input = self.input_layer_list[i](conv_input)
            conv_input = self.maxpool(conv_input)
            for run in range(self.num_rec):
                if run == 0:
                    current_batch_size, channels height,
                        width = conv_input.get_shape()
                    cell_state = K.zeros([current_batch_size, channels,
                        height, width])
                cell_state = self.cell_list[i](conv_input, cell_state)
            conv_input = cell_state
            # pool the features for classifier
            conv_input = self.maxpool(conv_input)
