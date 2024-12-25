from EDF_Part_B_and_C import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Constants
WIDTH = 20
CLASS1_SIZE = 50
CLASS2_SIZE = 50
N_FEATURES = 2
TEST_SIZE = 0.25
EPOCHS = 1000
LEARNING_RATE = 0.1
BATCH_SIZE = 5

# Define means and covariances
COV = np.array([[0.05, 0], [0, 0.05]])
MEANS = [np.array([0, 0]), np.array([1, 1]), np.array([1, 0]), np.array([0, 1])]

# Generate data
X = np.vstack([multivariate_normal.rvs(mean, COV, CLASS1_SIZE) for mean in MEANS])
y = np.hstack([np.zeros(CLASS1_SIZE * 2), np.ones(CLASS2_SIZE * 2)])

# Plot data
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Scatter Plot of Generated Data")
plt.show()

# Split data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
test_set_size = int(len(X) * TEST_SIZE)
X_train, X_test = X[indices[test_set_size:]], X[indices[:test_set_size]]
y_train, y_test = y[indices[test_set_size:]], y[indices[:test_set_size]]

# Create nodes
x_node = Input()
y_node = Input()

# Build computation graph
hidden_layer = Linear(x_node, WIDTH, N_FEATURES)
hidden_activation = Sigmoid(hidden_layer)
output_layer = Linear(hidden_activation, 1, WIDTH)
output_activation = Sigmoid(output_layer)
loss = BCE(y_node, output_activation)

# Graph and trainable parameter initialization
graph, trainables = [], []
def find_graph_and_trainables(last_node):
    graph.append(last_node)
    for node in graph:
        for inp in node.inputs:
            if inp not in graph:
                graph.append(inp)
            if inp.trainable:
                trainables.append(inp)

find_graph_and_trainables(loss)

# Forward and backward pass
def forward_pass():
    for node in reversed(graph):
        node.forward()

def backward_pass():
    for node in graph:
        node.backward()

# SGD update
def sgd_update():
    for param in trainables:
        param.value -= LEARNING_RATE * param.gradients[param].T

# Training loop
loss_per_epoch = []
for epoch in range(EPOCHS):
    epoch_loss = 0
    for i in range(0, X_train.shape[0], BATCH_SIZE):
        x_node.value = X_train[i:i + BATCH_SIZE]
        y_node.value = y_train[i:i + BATCH_SIZE].reshape(-1, 1)
        forward_pass()
        backward_pass()
        sgd_update()
        epoch_loss += loss.value

    avg_loss = epoch_loss / (X_train.shape[0] / BATCH_SIZE)
    loss_per_epoch.append(avg_loss)
    print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.6f}")

# Evaluate the model
correct_predictions = 0
for i in range(0, X_test.shape[0], BATCH_SIZE):
    x_node.value = X_test[i:i + BATCH_SIZE]
    y_node.value = y_test[i:i + BATCH_SIZE].reshape(-1, 1)
    forward_pass()
    predictions = (output_activation.value >= 0.5).astype(int)
    correct_predictions += (predictions.flatten() == y_test[i:i + BATCH_SIZE]).sum()

accuracy = correct_predictions / X_test.shape[0]
print(f"Accuracy: {accuracy * 100:.2f}%")

# Plot decision boundary
x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
Z = []
for i, j in zip(xx.ravel(), yy.ravel()):
    x_node.value = np.array([i, j]).reshape(1, -1)
    forward_pass()
    Z.append(output_activation.value.item())

Z = np.array(Z).reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision Boundary")
plt.show()
