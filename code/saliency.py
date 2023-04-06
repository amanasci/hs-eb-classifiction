import numpy as np
import tensorflow as tf 

# Define a function to compute the saliency map
@tf.function
def compute_saliency( input_data): #Use (input_data, params) for combo model
    with tf.GradientTape() as tape:
        # Watch the input variables
        tape.watch(input_data)
        
        # Compute the output of the model
        output = model_lstm_only(input_data) #change the model as per the model being used. 
        # Use model2(input_data, params) for the combo model.
        
    # Compute the gradient of the output with respect to the input
    gradients = tape.gradient(output, input_data)

    # Compute the saliency map using the gradient * input method
    saliency_map = input_data * gradients
    
    # Return the absolute value of the saliency map
    return tf.abs(saliency_map)


# Generate some test data
input_data = xl_train[:10] # xl_train is the training data for the light curves and the [:10] is the first 10 examples used as the one batch.

# Compute the saliency maps for the test data
saliency_maps = compute_saliency(input_data)

# Print the saliency maps for the first example in the batch
print(saliency_maps[0])