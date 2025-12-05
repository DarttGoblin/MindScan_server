import matplotlib.pyplot as plt  

epochs = list(range(1, 11))  

# Binary classification metrics
train_loss_bin = [0.1930, 0.0929, 0.0655, 0.0452, 0.0466, 0.0307, 0.0225, 0.0165, 0.0122, 0.0139]  
val_loss_bin   = [0.0891, 0.0796, 0.0619, 0.0580, 0.0841, 0.0438, 0.0416, 0.0336, 0.0274, 0.0683]  
train_acc_bin  = [0.9326, 0.9660, 0.9788, 0.9854, 0.9840, 0.9906, 0.9948, 0.9962, 0.9976, 0.9958]  
val_acc_bin    = [0.9681, 0.9778, 0.9792, 0.9792, 0.9750, 0.9833, 0.9847, 0.9903, 0.9903, 0.9806]  

# Multiclass classification metrics
train_loss_multi = [0.6788, 0.4188, 0.3318, 0.2955, 0.2587, 0.2193, 0.2051, 0.1838, 0.1587, 0.1346]  
val_loss_multi   = [0.5135, 0.4558, 0.3805, 0.3598, 0.3444, 0.3537, 0.3471, 0.3571, 0.3329, 0.3215]  
train_acc_multi  = [0.7088, 0.8331, 0.8712, 0.8831, 0.9062, 0.9194, 0.9262, 0.9306, 0.9344, 0.9538]  
val_acc_multi    = [0.7905, 0.8180, 0.8479, 0.8479, 0.8628, 0.8579, 0.8579, 0.8554, 0.8628, 0.8728]  

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
plot_and_save(epochs, train_loss_bin, val_loss_bin, 'Loss', 'Binary Classification Loss', 'binary_loss.png')
plot_and_save(epochs, train_acc_bin, val_acc_bin, 'Accuracy', 'Binary Classification Accuracy', 'binary_accuracy.png')

# Multiclass plots
plot_and_save(epochs, train_loss_multi, val_loss_multi, 'Loss', 'Multiclass Classification Loss', 'multi_loss.png')
plot_and_save(epochs, train_acc_multi, val_acc_multi, 'Accuracy', 'Multiclass Classification Accuracy', 'multi_accuracy.png')