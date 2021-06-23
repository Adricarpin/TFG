import matplotlib.gridspec as gridspec
from tensorflow import keras
import matplotlib.pyplot as plt

# Load model
model = keras.models.load_model("umCNN.h5")

filters, biases = model.layers[0].get_weights()

# normalize filter values to 0-1
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# Plot
plt.figure(figsize = (10,10))
gs1 = gridspec.GridSpec(10, 10)
gs1.update(wspace=0.025, hspace=0.15) # set the spacing between axes.

for i in range(100):
	f = filters[:, :, i]
	ax1 = plt.subplot(gs1[i])
	plt.imshow(f)
	plt.axis('off')
	ax1.set_xticklabels([])
	ax1.set_yticklabels([])
	ax1.set_aspect('equal')

plt.show()

