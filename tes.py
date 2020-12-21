import numpy as np
import h5py
import matplotlib.pyplot as plt
#%matplotlib inline
np.random.seed(1)

print("selesai 1")

def load_kaggle_dataset():
    train = np.load('train.npz')
    valid = np.load('valid.npz')
    train_x_original, train_y = train['X'], train['Y']
    valid_x_original, valid_y = valid['X'], valid['Y']
    return train_x_original, train_y, valid_x_original, valid_y

print("selesai 2")

def load_coursera_dataset():
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    train_x_original = np.array(train_dataset["train_set_x"][:], dtype='float') # your train set features
    train_y = np.array(train_dataset["train_set_y"][:], dtype='float') # your train set labels

    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    valid_x_original = np.array(test_dataset["test_set_x"][:], dtype='float') # your test set features
    valid_y = np.array(test_dataset["test_set_y"][:], dtype='float') # your test set labels

    train_y = train_y.reshape((1, train_y.shape[0]))
    valid_y = valid_y.reshape((1, valid_y.shape[0]))
    
    return train_x_original, train_y, valid_x_original, valid_y

print("selesai 3")

train_x_original, train_y, valid_x_original, valid_y = load_kaggle_dataset()
print("selesai 4")
train_x_original.shape, valid_x_original.shape, train_y.shape, valid_y.shape
print(train_x_original.shape)
print("\n")
print(valid_x_original.shape)
print("\n")
print(train_y.shape)
print("\n")
print(valid_y.shape)
print("selesai 5")

def image2vec(image_rgb_matrix):
    return image_rgb_matrix.reshape(image_rgb_matrix.shape[0], -1).T

print("selesai 6")
train_x = image2vec(train_x_original)
valid_x = image2vec(valid_x_original)
        
print ("sanity check after reshaping: " + str(train_x[0:5,0]))

""" Normalize our dataset """
train_x /= 255.
valid_x /= 255.

train_x.shape, valid_x.shape

print("selesai 7")

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

print("selesai 8")

def mean_squared_loss(A, Y):
    return (A - Y)**2

print("selesai 9")

def forward_propagate(X, W, b):
    
    """ Number of training samples """
    m = X.shape[1]
    
    """ We do the linear transformation of our entire dataset. (W.X + b) """
    linear_transformation = np.matmul(W.T, X) + b
    
    """ Apply sigmoid activation to bring outputs within range [0..1] """
    activation = sigmoid(linear_transformation)
    
    return activation

print("selesai 11")

def backpropagate(A, X, Y):
    # Y is of shape 1-by-m
    m = Y.shape[1]
    
    dY_hat = A - Y
    dA = dY_hat * sigmoid(A) * (1 - sigmoid(A))
    
    """ Average the value for dW and dB to get the mean learning signal """
    dW = (1/float(m)) * np.matmul(X, dA.T)
    db = (1/float(m)) * np.sum(dA)
    return dW, db

print("selesai 12")

def propagate(X, Y, W, b, is_eval=False):
    
    # Number of samples in the dataset.
    m = X.shape[1]
    
    # Forward propagation
    A = forward_propagate(X, W, b)
    
    """ Average loss over the entire dataset """
    loss = (1 / float(2*m)) * np.sum(mean_squared_loss(A, Y))
    
    if not is_eval:
        # Backpropagation
        dW, db = backpropagate(A, X, Y)
        return loss, dW, db
    return loss

print("selesai 13")

def model(train_X, train_Y, valid_X, valid_Y, W, B, epochs, learning_rate, print_interval=100):
    train_loss = []
    valid_loss = []
    
    for itr in range(epochs):
        """ Do one pass of forward and backward propagation on the training dataset."""
        training_loss, dW, dB = propagate(train_X, train_Y, W, B)
        train_loss.append(training_loss)
        
        """ SGD on the parameters """
        W = W - (learning_rate * dW)
        B = B - (learning_rate * dB)
        
        """ Evaluate on the validation dataset and record the loss"""
        validation_loss = propagate(valid_X, valid_Y, W, B, is_eval=True)
        valid_loss.append(validation_loss)
        
        """ Print after every `print_interval` intervals """
        if itr > 0 and itr % print_interval == 0:
            print("Epochs =", itr, "train loss =", training_loss, "and validation loss = ", validation_loss)
    
    params = {"w": W,
              "b": B}
    
    return train_loss, valid_loss, params

print("selesai 14")

def init(dim):
    W = np.zeros((dim, 1))
    B = 0.
    return W, B

print("selesai 15")

def predict(W, b, X, Y):
    """ Number of examples in the test set"""
    m = Y.shape[1]
    
    """ Forward propagation on the images in the test set """
    A = sigmoid(np.dot(W.T, X) + b)
    
    """ 
        Use the activation values to make a prediction by the model.
        If the activation is > 0.5, that means our model predicts class 1
        else class 0. That's just the way we have set up. We can 
        set it up any which way and train the model accordingly.
    """
    Y_prediction = A > 0.5
    
    """ Finally finding out how many predictions we got right """
    return np.sum(Y_prediction == Y) * 100 / m

print("selesai 16")

W, B = init(train_x.shape[0])
training_losses, validation_losses, params = model(train_x, train_y, valid_x, valid_y, W, B, 5000, 0.003, print_interval=500)

print("selesai 17")

W = params["w"]
B = params["b"]

# Print train/test Errors
print("Train accuracy: {} %".format(predict(W, B, train_x, train_y)))
print("Test accuracy: {} %".format(predict(W, B, valid_x, valid_y)))

print("selesai 17")

X = np.arange(0, 5000)
trainY = training_losses
validY = validation_losses
plt.figure(figsize=(10,7))
plt.title("Loss over epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(X, trainY, label="training loss")
plt.plot(X, validY, label="validation loss")
plt.legend(loc="best")
plt.show()

print("selesai 18")

import warnings
warnings.filterwarnings('ignore')
import matplotlib.image as mpimg
from skimage.transform import resize
path_to_image = "270.jpg"
custom_image = np.asarray(mpimg.imread(path_to_image))
custom_image_resized = resize(custom_image, (64, 64, 3))
custom_image_resized = np.expand_dims(custom_image_resized, axis=0)
custom_image_resized = image2vec(custom_image_resized) / 255.
activation = forward_propagate(custom_image_resized, W, B)
print("ACTIVATION = ")
print(activation)
plt.imshow(custom_image)
print("It's a cat" if activation > 0.5 else "It's a dog")

print("selesai 19")