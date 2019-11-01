import torch
import torch.nn as nn
import numpy as np

import m1Cell

def get_same_padding(image_size, filter_size, stride):
    padding_x = (image_size[0] * stride - 1 - image_size[0] + filter_size) / 2
    padding_y = (image_size[1] * stride - 1 - image_size[1] + filter_size) / 2

    return (np.int(padding_x), np.int(padding_y))

class M1Classifier(nn.Module):
    def __init__(self, kernels, input_channels, output_channels, stride,
    batch_size, image_size, num_rec, num_layers, out_nodes):
        super(M1Classifier, self).__init__()
        self.num_rec = num_rec
        # Unpack the kernels
        kernel_3, kernel_5, kernel_7 = kernels
        # unpack output nodes
        out_node1, num_classes = out_nodes
        # pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # padding for 3x3 filter
        padding_3 = get_same_padding(image_size, kernel_3, stride)
        #print(type(padding_3))
        #print(type(padding_3[0]))
        # padding for 5x5 filter
        padding_5 = get_same_padding(image_size, kernel_5, stride)
        # padding for 7x7 filter
        padding_7 = get_same_padding(image_size, kernel_7, stride)
        #print(padding_5)
        # First input layer stack
        self.input_layer_1 = nn.Conv2d(input_channels, output_channels,
            kernel_size=kernel_3, padding=padding_3)
        # Second input layer stack
        self.input_layer_2 = nn.Conv2d(input_channels, output_channels,
            kernel_size=kernel_5, padding=padding_5)
        # Third input layer stack
        self.input_layer_3 = nn.Conv2d(input_channels, output_channels,
            kernel_size=kernel_7, padding=padding_7)

        # Cell input and cell objects
        self.input_layer_list = []
        self.cell_list = []
        self.num_layers = num_layers
        for i in range(self.num_layers):
            self.input_layer_list.append(nn.Conv2d(output_channels*3,
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
        self.fc_1 = nn.Linear(in_features=output_channels*16*16,
            out_features=out_node1)
        #self.fc_2 = tf.keras.layers.Dense(out_node1)
        self.fc_2 = nn.Linear(in_features=out_node1,
            out_features=num_classes)

    def forward(self, u):
        # Compute the input layer output
        conv_input_1 = self.input_layer_1(u)
        conv_input_2 = self.input_layer_2(u)
        conv_input_3 = self.input_layer_3(u)

        # concatenate the computed features
        conv_input = torch.cat([conv_input_1, conv_input_2, conv_input_3],
            axis=1)
        conv_input = self.maxpool(conv_input)
        #conv_input = self.maxpool_2d(conv_input)
        for i in range(self.num_layers):
            conv_input = self.input_layer_list[i](conv_input)
            conv_input = self.maxpool(conv_input)
            for run in range(self.num_rec):
                if run == 0:
                    current_batch_size, channels, height, width = conv_input.size()
                    cell_state = torch.zeros([current_batch_size, channels,
                        height, width])
                cell_state = self.cell_list[i](conv_input, cell_state)
            conv_input = cell_state
            # pool the features for classifier
            conv_input = self.maxpool(conv_input)

        # expand features
        conv_input = self.expand_features(conv_input)
        conv_input = self.maxpool(conv_input)
        # flatten the cell state
        flat_out = self.flat_layer(conv_input)
        #print(conv_input.size())
        # Dropout
        drop = self.drp(flat_out)
        # Classifier layers
        fc_out = torch.tanh(self.fc_1(drop))
        class_vect = self.fc_2(fc_out)

        return class_vect
'''
inputs = torch.empty(1, 3, 224, 224).uniform_(0, 1)
print(inputs)

kernels = [3, 5, 7]
output_channels = 10
input_channels = 3
stride = 1
batch_size = 1
image_size = [224, 224]
num_rec = 2
num_layers = 1
out_nodes = [100, 10]

model = M1Classifier(kernels, input_channels, output_channels, stride,
batch_size, image_size, num_rec, num_layers, out_nodes)

out = model(inputs)
print(out)
'''
