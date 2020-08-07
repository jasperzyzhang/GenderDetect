import matplotlib.pyplot as plt
plt.style.use('ggplot')

def visualize(hist,path):
    # Plot loss function value through epochs
    plt.figure(figsize=(18, 4))
    plt.plot(hist.history['loss'], label='train')
    plt.plot(hist.history['val_loss'], label='valid')
    plt.legend()
    plt.title('Traininig using Celeba and Simulated Mask Face dropout 0.5 Loss Function')
    plt.show()
    plt.savefig(path + 'lossfunction.png')

    # Plot accuracy through epochs
    plt.figure(figsize=(18, 4))
    plt.plot(hist.history['accuracy'], label='train')
    plt.plot(hist.history['val_accuracy'], label='valid')
    plt.legend()
    plt.title('Traininig using Celeba and Simulated Mask Face Drop out 0.5 Accuracy')
    plt.show()
    plt.savefig(path + 'Acc.png')