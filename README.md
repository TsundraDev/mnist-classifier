# mnist-classifier

This repo contains a custom Neural Network designed to classify images provided from the MNIST dataset.

It is a means to showcase basic principles from Neural Networks.

## Architecture

The Neural Network takes in a 28*28 grey-scaled image of a single digit number and attempts to predict the written number.
There are 2 hidden layers of the network whose size can be configured.
The output layer contains a probability distribution of what the network believes the number to be written

The first hidden layer is calculated by multiplying the inputs by a weight matrix.
The result is then passed through an activation function.
For this example, we've chosen the tanh function.

The second hidden layer functions similarly as the first layer, using the output from the first layer as input.

Finally, the output is calculated by multiplying the output of the second layer by a final weight matrix and passed through the softmax function, used to turn any distribution into a probability distribution.

## Training

The training method used is gradient descent. This method attempts to find the minimum of a function by calculating the gradient and following along its path to its minimum. The main strategy used in training is to have a loss function, which functions similarly as an error function, and to minimise it by modifying the weights.

In our example, we use the cross-entropy function due to its ease of calculating the gradient when used in conjunction with the softmax layer.
The cross-entropy function takes in a batch of inputs and expected outputs, as well as the weights used in the model. Using the gradient descent method, the cross-entropy is minimised according to the weights.

### Noisy training

During training, we add a bit of noise to each layer when performing calculations. This is an attempt to avoid over-fitting as similar inputs (differed by little noise) should roughly give the same output.

### Batch training

Ideally, training should be done on all training data at the same time, the gradient being calculated for all the data at the same time. This method is however quite expensive. As such, a compromise is to use only a subset aka. batch of the training data and to train on multiple batches. An assumption is made that each batch roughly corresponds to the same data as the overall dataset.

Testing is made after all batches have been run to measure the effectiveness of the network.
A run of all batches is called an epoch, and we can measure the effectiveness of the network at each epoch.
