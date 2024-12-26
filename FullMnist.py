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
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)
# Define constants
WIDTH = 512  # Hidden layer width
N_FEATURES = 784  # Number of input features (28x28 images)
N_OUTPUT = 10  # Number of output classes
TEST_SIZE = 0.4  # Test set proportion
EPOCHS = 5  # Number of epochs
LEARNING_RATE = 0.001  # Learning rate
BATCH_SIZE = 128  # Batch size
INITIALIZATION_SCALE = 0.01  # Scale for weight initialization

# Create nodes for the computational graph
x_node = Input()
y_node = Input()

# Build the network structure
u_node1 = Linear(x_node, output_size=WIDTH, input_size=N_FEATURES, initialization_controller=INITIALIZATION_SCALE)
H1_node = ReLU(u_node1)

u_node2 = Linear(H1_node, output_size=WIDTH, input_size=WIDTH, initialization_controller=INITIALIZATION_SCALE)
H2_node = ReLU(u_node2)

u_node3 = Linear(H2_node, output_size=N_OUTPUT, input_size=WIDTH, initialization_controller=INITIALIZATION_SCALE)
H3_node = Softmax(u_node3)
# Loss function
loss = CE(y_node, H3_node)
graph=[]
trainable=[]
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
# SGD Update
def sgd_update(trainables, learning_rate):
    for t in trainables:
        t.value -= learning_rate * t.gradients[t].T
loss_per_epoch=[]
graph=[]
trainable=[]
find_graph_and_trainable(loss,graph,trainable)
for epoch in range(EPOCHS):
    loss_value = 0
    for i in range(0, X_train_flattened.shape[0], BATCH_SIZE):
        x_node.value = X_train_flattened[i:i + BATCH_SIZE]
        y_node.value = y_train[i:i + BATCH_SIZE]
        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainable, LEARNING_RATE)
        loss_value += loss.value
    avg_loss = loss_value / (X_train_flattened.shape[0] / BATCH_SIZE)
    loss_per_epoch.append(avg_loss)
    print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.6f}")

# Evaluate the model

correct_predictions = 0
for i in range(0, X_test_flattened.shape[0], BATCH_SIZE):
    x_batch = X_test_flattened[i:i + BATCH_SIZE]
    y_batch = y_test[i:i + BATCH_SIZE]
    x_node.value = x_batch
    y_node.value = y_batch
    forward_pass(graph)
    for k in range(len(x_batch)):
        predicted_class = np.argmax(H3_node.value[k])
        true_class = y_batch[k]
        if predicted_class == true_class:
            correct_predictions += 1
accuracy = correct_predictions / X_test_flattened.shape[0]
print(f"Accuracy: {accuracy * 100:.02f}%")