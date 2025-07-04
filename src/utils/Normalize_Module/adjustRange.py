def adjustRange(input_array, input_range, output_range):

    norm_array = (input_array - input_range[0]) / (input_range[1] - input_range[0])

    output_array = (norm_array * (output_range[1] - output_range[0])) + output_range[0]

    return output_array