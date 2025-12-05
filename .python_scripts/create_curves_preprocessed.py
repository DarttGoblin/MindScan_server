import matplotlib.pyplot as plt  

epochs = list(range(1, 11))  

# Binary classification metrics
train_loss_bin = [0.2308, 0.1100, 0.0861, 0.0590, 0.0448, 0.0372, 0.0274, 0.0205, 0.0163, 0.0105]
val_loss_bin   = [0.1341, 0.1098, 0.1159, 0.0601, 0.0568, 0.0745, 0.0556, 0.0518, 0.0568, 0.0485]
train_acc_bin  = [0.9111, 0.9618, 0.9750, 0.9795, 0.9861, 0.9892, 0.9924, 0.9955, 0.9969, 0.9976]
val_acc_bin    = [0.9444, 0.9653, 0.9639, 0.9819, 0.9819, 0.9722, 0.9833, 0.9875, 0.9875, 0.9903]

# Multiclass classification metrics
train_loss_multi = [0.6899, 0.4112, 0.3542, 0.2958, 0.2482, 0.2163, 0.2095, 0.1936, 0.1499, 0.1377]
val_loss_multi   = [0.4399, 0.3584, 0.3068, 0.2756, 0.2628, 0.2618, 0.2430, 0.2733, 0.2218, 0.2334]
train_acc_multi  = [0.6938, 0.8394, 0.8487, 0.8831, 0.9000, 0.9225, 0.9212, 0.9287, 0.9550, 0.9519]
val_acc_multi    = [0.7930, 0.8728, 0.8853, 0.9052, 0.9077, 0.9027, 0.9102, 0.9027, 0.9102, 0.9127]

# Function to plot and save
def plot_and_save(epochs, train, val, ylabel, title, filename):
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train, label='Training ' + ylabel)
    plt.plot(epochs, val, label='Validation ' + ylabel)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()

# Binary plots
plot_and_save(epochs, train_loss_bin, val_loss_bin, 'Loss', 'Binary Classification Loss', 'binary_loss_preprocessed.png')
plot_and_save(epochs, train_acc_bin, val_acc_bin, 'Accuracy', 'Binary Classification Accuracy', 'binary_accuracy_preprocessed.png')

# Multiclass plots
plot_and_save(epochs, train_loss_multi, val_loss_multi, 'Loss', 'Multiclass Classification Loss', 'multi_loss_preprocessed.png')
plot_and_save(epochs, train_acc_multi, val_acc_multi, 'Accuracy', 'Multiclass Classification Accuracy', 'multi_accuracy_preprocessed.png')