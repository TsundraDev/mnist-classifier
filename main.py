from utils import read_file_img, read_file_label
from NeuralNetwork import NeuralNetwork

import matplotlib.pyplot as plt

# Data files
train_image_file = "data/train-images.idx3-ubyte"
train_label_file = "data/train-labels.idx1-ubyte"
test_image_file  = "data/t10k-images.idx3-ubyte"
test_label_file  = "data/t10k-labels.idx1-ubyte"

train_image_bank = read_file_img(train_image_file)
train_label_bank = read_file_label(train_label_file)
test_image_bank  = read_file_img(test_image_file)
test_label_bank  = read_file_label(test_label_file)


# Limit training data
train_image = train_image_bank
train_label = train_label_bank
test_image = test_image_bank
test_label = test_label_bank

nn = NeuralNetwork(28*28, 20, 12, 10)

# Initial test
print("Initial test")
test = [nn.test(test_image, test_label)]
nn.test(train_image, train_label)
print("---")

# Learning
max_epoch = 10
batch_size = 64
for epoch in range(max_epoch):
    for i in range(0, len(train_image), batch_size):
        nn.backward(train_image[i:i+batch_size], train_label[i:i+batch_size])
    test += [nn.test(test_image, test_label)]
    


# Final test
print("---")
nn.test(test_image, test_label)
nn.test(train_image, train_label)

test = [100 * t for t in test]
plt.plot(test)
plt.show()


