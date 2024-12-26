from keras.datasets import mnist
from matplotlib import pyplot
from EDF_Part_B_and_C import *

#loading the dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#printing the shapes of the vectors
print('X_train: ' + str(X_train.shape))
print('Y_train: ' + str(y_train.shape))
print('X_test:  '  + str(X_test.shape))
print('Y_test:  '  + str(y_test.shape))
for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(X_train[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()
# Normalize and reshape data
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32')
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32')

# Define constants
N_OUTPUT = 10
epochs = 2
learning_rate = 0.015
batch_size = 64

# Create nodes
x_node = Input()
x_node.value = np.zeros((batch_size, 28, 28, 1))
y_node = Input()
# Add convolutional layers
Conv1 = Conv(x_node, 1, 16,3,1,1)  # Input has one channel
u_node1 = ReLU(Conv1)
pooling_node1 = MaxPooling(u_node1)
Conv2 = Conv(pooling_node1, 16, 32,3,1,1)
u_node2 = ReLU(Conv2)
pooling_node2 = MaxPooling(u_node2)
Conv3 = Conv(pooling_node2, 32, 64 ,3,1,1)
u_node3 = ReLU(Conv3)
pooling_node3 = MaxPooling(u_node3)
Conv4 = Conv(pooling_node3, 64, 128 ,3,1,1)
u_node4 = ReLU(Conv4)
pooling_node4 = MaxPooling(u_node4)
# Flatten node
flatten_node = Flatten(pooling_node4)
# Fully connected layer
v_node = Linear(flatten_node, output_size=10, input_size=128, initialization_controller=0.005)
#last_node = ReLU(v_node)
y_pre = Softmax(v_node)
# Loss function
loss = CE(y_node, y_pre)
graph=[]
trainable=[]
loss_per_epoch=[]
def find_graph_and_trainable(last_node, graph , trainables ):
    graph.append(last_node)
    for n in graph :
        for t in n.inputs :
            if not t.visited :
                graph.append(t)
                t.visited=True
            if t.trainable :
                trainables.append(t)
def forward_pass(graph):
    for n in graph[::-1]:
        n.forward()
def backward_pass(graph):
    for n in graph:
        n.backward()
def sgd_update(trainables, learning_rate):
    for t in trainables:
        if type(t) is not Conv :
            t.value -= learning_rate * t.gradients[t].T
        else:
            t._w -= learning_rate * t.gradients['weights']
            t._b -= learning_rate * t.gradients['bias']
find_graph_and_trainable(loss,graph,trainable)
for epoch in range(epochs):
    loss_value = 0
    for i in range(0, X_train.shape[0], batch_size):
        x_node.value = X_train[i:i + batch_size]
        y_node.value = y_train[i:i + batch_size]
        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainable, learning_rate)
        loss_value += loss.value
        avg_loss = loss_value / (X_train.shape[0] / batch_size)
        print(loss.value)
        loss_per_epoch.append(avg_loss)
    print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.6f}")

# Evaluate the model

correct_predictions = 0
for i in range(0, X_test.shape[0], batch_size):
    x_batch = X_test[i:i + batch_size]
    y_batch = y_test[i:i + batch_size]
    x_node.value = x_batch
    y_node.value = y_batch
    forward_pass(graph)
    for k in range(len(x_batch)):
        predicted_class = np.argmax(y_pre.value[k])
        true_class = y_batch[k]
        if predicted_class == true_class:
            correct_predictions += 1
accuracy = correct_predictions / X_test.shape[0]
print(f"Accuracy: {accuracy * 100:.02f}%")