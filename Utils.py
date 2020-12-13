"""
/*
 * @author: AhmedSalah
 */
########################################################################
Utils.py file is used to prepare the chess model
contains the following functions:
    --add_conv(model, filters_arr, input_shape, kernel_size=3, pool_size=2,
             strides=2, flatten=True, activation='relu')

        used to add convolution layers to the sequential model

    --add_ann(model, layers_settings, hidden_activation='relu',
            output_activation='softmax')

        used to add layers to the sequential model

    --prepare_model(model, filters_arr, layers_settings, input_shape=[8, 8, 12],kernel_size=3,
        pool_size=2, strides=2, flatten=True, conv_activation='relu', hidden_activation='relu', output_activation = 'softmax')

        integrates between add_conv() and add_ann() to build a combinrational model
"""
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
#**********************************************************************#
def add_conv(model, filters_arr, input_shape, kernel_size=3, pool_size=2,
             strides=2, flatten=True, activation='relu'):
    """
    Argument:
    model -- model of type tensorflow.keras.models.Sequential
    filter_arr -- array of integer values corresponding to the filter at each
                    convolution layer
    input_shape -- array of integer values representing the shape of the input features
    kernel_size -- size of the kernel mask, default = 3 representing a shape of 3X3
    pool_size -- size of the pool mask over the kernel, default = 2 representing a shape of 2X2
    strides -- steps of the mocing pool, default = 2
    flatten -- flatten the tensor into a vector of shape (product of the features,1), default = True
    activation -- activation function, default = 'relu'
    
    Returns:
    model -- model of type tensorflow.keras.models.Sequential
    """
    # --------Convolution and max Pooling--------#
    num_conv_layers = len(filters_arr)
    # first layer
    model.add(Conv2D(filters=filters_arr[0], kernel_size=kernel_size, activation=activation, input_shape=input_shape))
    model.add(MaxPool2D(pool_size=pool_size, strides=strides))
    # subsequent layers
    for i in range(1, num_conv_layers):
        model.add(Conv2D(filters=filters_arr[i], kernel_size=kernel_size, activation=activation))
        model.add(MaxPool2D(pool_size=pool_size, strides=strides))

    # --------Flattening Outputs--------#
    model.add(Flatten())

    return model

def add_ann(model, layers_settings, hidden_activation='relu',
            output_activation='softmax'):
    """
    Argument:
    model -- model of type tensorflow.keras.models.Sequential
    layers_settings -- array of integer values corresponding to the number of nodes in each layer
                        ex: layers_settings = [2,10,5] represents 3 layers, layer 1 has 2 nodes
                            layer 2 has 10 nodes, and layer 3 has 5 nodes
    hidden_activation -- activation function of the hidden layers, default = 'relu'
    output_activation -- activation function of the output layer, default = 'softmax'

    Returns:
    model -- model of type tensorflow.keras.models.Sequential
    """
    # --------hidden_layers--------#
    num_hidden_layers = len(layers_settings)
    for i in range(num_hidden_layers - 1):
        model.add(Dense(units=layers_settings[i], activation=hidden_activation))

    # --------Output layer--------#
    model.add(Dense(units=layers_settings[-1], activation=output_activation))

    return model

def prepare_model(model, filters_arr, layers_settings, input_shape=[8, 8, 12],kernel_size=3,\
    pool_size=2, strides=2, flatten=True, conv_activation='relu', hidden_activation='relu', output_activation = 'softmax'):
    """
    Argument:
    model -- model of type tensorflow.keras.models.Sequential
    filter_arr -- array of integer values corresponding to the filter at each
                    convolution layer
    layers_settings -- array of integer values corresponding to the number of nodes in each layer
                        ex: layers_settings = [2,10,5] represents 3 layers, layer 1 has 2 nodes
                            layer 2 has 10 nodes, and layer 3 has 5 nodes
    input_shape -- array of integer values representing the shape of the input features
    kernel_size -- size of the kernel mask, default = 3 representing a shape of 3X3
    pool_size -- size of the pool mask over the kernel, default = 2 representing a shape of 2X2
    strides -- steps of the mocing pool, default = 2
    flatten -- flatten the tensor into a vector of shape (product of the features,1), default = True
    conv_activation -- activation function, default = 'relu'
    hidden_activation -- activation function of the hidden layers, default = 'relu'
    output_activation -- activation function of the output layer, default = 'softmax'

    Returns:
    model -- model of type tensorflow.keras.models.Sequential
    """
    # --------Convolution layers--------#
    if filters_arr:
        add_conv(model, filters_arr, input_shape, kernel_size=kernel_size,
                 pool_size=pool_size, strides=strides, flatten=True, activation=conv_activation)
    # --------Hidden and Output layers--------#
    add_ann(model, layers_settings,hidden_activation=hidden_activation,
        output_activation  = output_activation)
    # --------Setting Optimizer and finalizing the model--------#
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics='accuracy')

    return model
