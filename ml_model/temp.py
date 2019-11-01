import tensorflow as tf
import tensorflow.keras.backend as K

tf.enable_eager_execution()

class m1Cell(tf.keras.Model):
    def __init__(self, kernels, output_channels, dialation):
        super(m1Cell, self).__init__(name="")
        # unpack the kernels
        kernel1, kernel2, kernel3 = kernels
        # unpack the channels (output)
        ch_rec_layer = output_channels[0]
        # Kernel initializer method
        kernel_init = "glorot_normal"
        # A1 * x layer
        self.conv_1 = tf.keras.layers.Conv2D(ch_rec_layer, (kernel1, kernel1), dilation_rate=(dialation, dialation),
                                            padding="same", kernel_initializer=kernel_init)
        # A2 * x layer
        self.conv_2 = tf.keras.layers.Conv2D(ch_rec_layer, (kernel2, kernel2), dilation_rate=(dialation, dialation),
                                            padding="same", kernel_initializer=kernel_init)
        # A3 * x layer
        self.conv_3 = tf.keras.layers.Conv2D(ch_rec_layer, (kernel3, kernel3), dilation_rate=(dialation, dialation),
                                            padding="same", kernel_initializer=kernel_init)

        # B1 * x² layer
        self.conv_xx_1 = tf.keras.layers.Conv2D(ch_rec_layer, (kernel1, kernel1), dilation_rate=(dialation, dialation),
                                                padding="same", kernel_initializer=kernel_init)
        # B2 * x² layer
        self.conv_xx_2 = tf.keras.layers.Conv2D(ch_rec_layer, (kernel2, kernel2), dilation_rate=(dialation, dialation),
                                                padding="same", kernel_initializer=kernel_init)
        # B3 * x² layer
        self.conv_xx_3 = tf.keras.layers.Conv2D(ch_rec_layer, (kernel3, kernel3), dilation_rate=(dialation, dialation),
                                                padding="same", kernel_initializer=kernel_init)

        # gamma_1 * (A_1 * x + B_1 * x^2 + C_1 * u)
        self.conv_g_1 = tf.keras.layers.Conv2D(ch_rec_layer, (kernel1, kernel1), dilation_rate=(dialation, dialation),
                                                padding="same", kernel_initializer=kernel_init)
        # gamma_2 * (A_2 * x + B_2 * x^2 + C_2 * u)
        self.conv_g_2 = tf.keras.layers.Conv2D(ch_rec_layer, (kernel2, kernel2), dilation_rate=(dialation, dialation),
                                                padding="same", kernel_initializer=kernel_init)
        # gamma_3 * (A_3 * x + B_3 * x^2 + C_3 * u)
        self.conv_g_3 = tf.keras.layers.Conv2D(ch_rec_layer, (kernel3, kernel3), dilation_rate=(dialation, dialation),
                                                padding="same", kernel_initializer=kernel_init)

        # alpha_1 * x
        self.conv_a_1 = tf.keras.layers.Conv2D(ch_rec_layer, (kernel1, kernel1), dilation_rate=(dialation, dialation),
                                                padding="same", kernel_initializer=kernel_init)
        # alpha_2 * x²
        self.conv_a_2 = tf.keras.layers.Conv2D(ch_rec_layer, (kernel2, kernel2), dilation_rate=(dialation, dialation),
                                                padding="same", kernel_initializer=kernel_init)
        # alpha_3 * x³
        self.conv_a_3 = tf.keras.layers.Conv2D(ch_rec_layer, (kernel3, kernel3), dilation_rate=(dialation, dialation),
                                                padding="same", kernel_initializer=kernel_init)

    def __call__(self, inputs, state):
        # compute state multiplications
        #xx = tf.keras.layers.multiply([state, state])
        #xxx = tf.keras.layers.multiply([xx, state])
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
        f1 = K.tanh(tf.keras.layers.Add()([conv1, convxx1, inputs]))
        # Second function
        f2 = K.tanh(tf.keras.layers.Add()([conv2, convxx2, inputs]))
        # Third function
        f3 = K.tanh(tf.keras.layers.Add()([conv3, convxx3, inputs]))

        # Conv gamma_1
        convg1 = self.conv_g_1(f1)
        # Conv gamma_2
        convg2 = self.conv_g_2(f2)
        # Conv gamma_3
        convg3 = self.conv_g_3(f3)

        # Get the nonlinear function
        nonlinear = tf.keras.layers.Add()([convg1, convg2, convg3])

        # Get the non-activation convolutions
        # alpha_1 * x
        n_out_1 = self.conv_a_1(state)
        # alpha_2 * x²
        n_out_2 = self.conv_a_2(xx)
        # alpha_3 * x³
        n_out_3 = self.conv_a_3(xxx)

        # Compute the final state
        state = tf.keras.layers.Add()([n_out_1, n_out_2, n_out_3, nonlinear])

        return state
