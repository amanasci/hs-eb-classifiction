import matplotlib.pyplot as plt
import numpy as np

# Define a function to compute the attention weights for a given input and parameters
def compute_attention_weights(input_data):
    # Compute the attention weights using the model
    attention_weights = model_lstm_only.get_layer('lstm')(input_data)
    
    return attention_weights.numpy()

# Generate some test data
input_data = xl_train[:10]

# Compute the attention weights for the test data
attention_weights = compute_attention_weights(input_data)

# Plot the attention weights for a particular example in the batch
fig, ax = plt.subplots()
im = ax.imshow(attention_weights[3,:,:], cmap='coolwarm', aspect='auto')

# Add a colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Attention Weights', rotation=-90, va="bottom")

# Set the x and y axis labels
ax.set_xlabel('Input sequence')
ax.set_ylabel('LSTM output')

# Set the title of the plot
ax.set_title('Attention Weights for Example 1')

# Show the plot
plt.show()
