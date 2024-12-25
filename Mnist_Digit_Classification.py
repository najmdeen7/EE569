from EDF_Part_B_and_C import *
import numpy as np
from sklearn import datasets

# Define constants
HIDDEN_LAYER_WIDTH = 64
N_FEATURES = 64
N_CLASSES = 10
TEST_SIZE = 0.4
EPOCHS = 100
LEARNING_RATE = 0.005
BATCH_SIZE = 32

# Load dataset
mnist = datasets.load_digits()
X, y = mnist['data'], mnist['target'].astype(int)


# Shuffle and split data manually
def manual_train_test_split(X, y, test_size=0.4, random_seed=25):
    """
    Shuffle and split data into training and testing sets.
    """
    np.random.seed(random_seed)  # Set random seed for reproducibility
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    test_set_size = int(len(X) * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test


# Perform manual train-test split
X_train, X_test, y_train, y_test = manual_train_test_split(X, y, test_size=TEST_SIZE, random_seed=25)

# Create nodes
x_node = Input()
y_node = Input()
weight_init_std = 0.1  # Standard deviation for weight initialization

# Build computation graph
hidden_layer = Linear(x_node, HIDDEN_LAYER_WIDTH, N_FEATURES, weight_init_std)
hidden_activation = Sigmoid(hidden_layer)
output_layer = Linear(hidden_activation, N_CLASSES, HIDDEN_LAYER_WIDTH, weight_init_std)
output_activation = Softmax(output_layer)
loss_node = CE(y_node, output_activation)

# Graph management functions
graph = []
trainable = []


def find_graph_and_trainable(last_node, graph, trainables):
    """
    Build the computation graph and find trainable parameters.
    """
    graph.append(last_node)
    for node in graph:
        for input_node in node.inputs:
            if not input_node.visited:
                graph.append(input_node)
                input_node.visited = True
            if input_node.trainable:
                trainables.append(input_node)


def forward_pass(graph):
    """
    Perform a forward pass through the computation graph.
    """
    for node in reversed(graph):
        node.forward()


def backward_pass(graph):
    """
    Perform a backward pass through the computation graph.
    """
    for node in graph:
        node.backward()


def sgd_update(trainables, learning_rate=1e-2):
    """
    Update the trainable parameters using SGD.
    """
    for param in trainables:
        param.value -= learning_rate * param.gradients[param].T


# Training loop
loss_per_epoch = []
find_graph_and_trainable(loss_node, graph, trainable)

for epoch in range(EPOCHS):
    loss_value = 0
    for i in range(0, X_train.shape[0], BATCH_SIZE):
        # Batch data
        x_node.value = X_train[i:i + BATCH_SIZE]
        y_node.value = y_train[i:i + BATCH_SIZE]

        # Forward and backward pass
        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainable, LEARNING_RATE)

        # Accumulate loss
        loss_value += loss_node.value

    # Compute average loss
    avg_loss = loss_value / (X_train.shape[0] / BATCH_SIZE)
    loss_per_epoch.append(avg_loss)
    print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.6f}")

# Evaluation
correct_predictions = 0
for i in range(0, X_test.shape[0], BATCH_SIZE):
    x_batch = X_test[i:i + BATCH_SIZE]
    y_batch = y_test[i:i + BATCH_SIZE]
    x_node.value = x_batch
    y_node.value = y_batch
    forward_pass(graph)

    # Prediction and accuracy check
    for k in range(len(x_batch)):
        predicted_class = np.argmax(output_activation.value[k])
        true_class = y_batch[k]
        if predicted_class == true_class:
            correct_predictions += 1

accuracy = correct_predictions / X_test.shape[0]
print(f"Accuracy: {accuracy * 100:.02f}%")