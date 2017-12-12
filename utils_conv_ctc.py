def conv_output_length(input_length, filter_size, padding, stride,
                       dilation=1):
    ''' Compute the length of the output sequence after 1D convolution along
        time.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        padding (str): Only support `SAME` or `VALID`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    '''
    if input_length is None:
        return None
    assert padding in {'SAME', 'VALID'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if padding == 'SAME':
        output_length = input_length
    elif padding == 'VALID':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride


# Convolution Layer
#layer_1 = tf.contrib.layers.convolution2d(inputs=inputs, num_outputs=num_hidden,
        #kernel_size=[5], stride=1, padding='VALID',
        #normalizer_fn=tf.contrib.layers.batch_norm,
        #normalizer_params={'is_training': True})


#output_lengths = conv_output_length(seq_len, 5, 'VALID', 1)
