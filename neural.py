import numpy as np
import struct as st
class NeuralNetwork(object):
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.depth = len(self.dimensions) - 1
        self.weights = []
        self.biases = []
        for j in range(self.depth):
            self.weights.append(np.random.normal(0.0, 1.0, (self.dimensions[j+1], self.dimensions[j])))
            self.biases.append(np.random.normal(0.0, 1.0, (self.dimensions[j+1], 1)))
    def sigmoid(self, x, derivative = False):
        return 1.0 / (np.exp(x) + np.exp(-x) + 2.0) if derivative else 1.0 / (1.0 + np.exp(-x))
    def computeKnots(self, data):
        """Computes all knots in the neural network with the current weights and returns the outcome"""
        if len(data) != self.dimensions[0]:
            print("Data has to have length " + str(self.dimensions[0]) + " for this neural network")
            return
        self.knots = []
        self.z_values = []
        temp = np.array(data)
        temp.shape = (self.dimensions[0], 1)
        self.knots.append(temp)
        for j in range(1, self.depth + 1):
            output = self.weights[j - 1].dot(self.knots[j-1])
            output.shape = (self.dimensions[j], 1)
            output += self.biases[j - 1]
            self.z_values.append(output)
            self.knots.append(self.sigmoid(output))
        return self.knots[self.depth]
    def gradientDescent(self, dataInput, dataOutput, sample_size = 20, num_rounds = 50, learning_rate = 3.0):
        if len(dataInput[0]) != self.dimensions[0]:
            print("The input data has to have length " + str(self.dimensions[0]) + " for this neural network")
            return
        if len(dataOutput[0]) != self.dimensions[self.depth]:
            print("The output data has to have length " + str(self.dimensions[self.depth]) + " for this neural network")
            return

        data_length = len(dataInput)
        num_batches = int(data_length / sample_size)
        for this_round in range(num_rounds):
            print(str(num_rounds-this_round))
            randomize = np.random.permutation(data_length)
            input_random = dataInput
            ouput_random = dataOutput
            error = 0
            for j in range(data_length):
                input_random[j] = dataInput[randomize[j]]
                ouput_random[j] = dataOutput[randomize[j]]
            for batch in range(num_batches):
                diffB = []
                diffW = []
                for level in range(self.depth):
                    diffW.append(np.zeros(np.shape(self.weights[level])))
                    diffB.append(np.zeros(np.shape(self.biases[level])))
                for sample in range(sample_size):
                    goal = np.array(ouput_random[sample_size * batch + sample])
                    goal.shape = (self.dimensions[self.depth], 1)
                    result = self.computeKnots(input_random[sample_size * batch + sample])
                    difference = result - goal
                    dCdy = 2.0 * difference
                    error += np.dot(np.transpose(difference), difference)
                    for level in range(self.depth - 1, -1, -1):
                        a = np.multiply(dCdy, self.sigmoid(self.z_values[level], True))
                        dCdy = np.matmul(np.transpose(self.weights[level]), a)
                        diffW[level] += np.matmul(a, np.transpose(self.knots[level]))
                        diffB[level] += a
                self.weights -= np.multiply(learning_rate / sample_size, diffW)
                self.biases -= np.multiply(learning_rate / sample_size, diffB)

            print(str(error/data_length))            
def loadData():
    def read_idx(filename):
        with open(filename, 'rb') as f:
            zero, data_type, dims = st.unpack('>HBB', f.read(4))
            shape = tuple(st.unpack('>I', f.read(4))[0] for d in range(dims))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

    labels_temp = read_idx("training_set/train-labels.idx1-ubyte")
    images_temp = read_idx("training_set/train-images.idx3-ubyte")
    labels = []
    images = []
    for j in range(len(labels_temp)):
        label = np.zeros(10)
        label[labels_temp[j]] = 1.0
        labels.append(label)
        images.append(np.ndarray.flatten(images_temp[j]) / 255)
    return images, labels

"""read the mnist database. use the first 50k samples as training data, the last 10k as test data"""
images, labels = loadData()
training_images, training_labels = images[0:49999], labels[0:49999]
test_images, test_labels = images[50000:59999], labels[50000:59999]

"""create a neural network with 28 * 28 input neurons, one hidden layer with 30 neurons, and an output layer with 10 neurons"""
myNet = NeuralNetwork([28 * 28, 30, 10])

"""use the training data to optimise the (initially random) weights and biases of the neural network"""
myNet.gradientDescent(training_images, training_labels, sample_size=10, num_rounds=10)

"""use the improved neural network on the test data. compare the computed results with the given labels and count how often the network was right"""
hits=0
numbers = np.zeros(10)
for j in range(len(test_images)):
    output = myNet.computeKnots(test_images[j])
    result = np.argmax(output)
    numbers[result] += 1
    if np.argmax(test_labels[j]) == result:
        hits += 1

print(str(hits / len(test_images)))
print(numbers)

np.set_printoptions(formatter={"all": lambda x: "%.3f" % x})

for j in range(len(myNet.weights)):
   myNet.weights[j] = np.round(myNet.weights[j], 3)
   myNet.biases[j] = np.round(myNet.biases[j], 3)

file_handle = open("weights.txt", "w")
file_handle.write("w0 = " + str(myNet.weights[0].tolist()) + "\n")
file_handle.write("w1 = " + str(myNet.weights[1].tolist()) + "\n")
file_handle.write("b0 = " + str(myNet.biases[0].tolist()) + "\n")
file_handle.write("b1 = " + str(myNet.biases[1].tolist()) + "\n")
file_handle.close()