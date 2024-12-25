
from EDF_Part_A import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Define constants
CLASS1_SIZE = 100
CLASS2_SIZE = 100
N_FEATURES = 2
N_OUTPUT = 1
TEST_SIZE = 0.25
epochs = 100
learning_rate = 0.05


# Define the means and covariances of the two components
MEAN1 = np.array([1, 0])
COV1 = np.array([[0.1, 0], [0, 0.1]])
MEAN2 = np.array([0,0])
COV2 = np.array([[0.1, 0], [0, 0.1]])

# Generate random points from the two components
X1 = multivariate_normal.rvs(MEAN1, COV1, CLASS1_SIZE)
X2 = multivariate_normal.rvs(MEAN2, COV2, CLASS2_SIZE)

# Combine the points and generate labels
X = np.vstack((X1, X2))
y = np.hstack((np.zeros(CLASS1_SIZE), np.ones(CLASS2_SIZE)))

# Plot the generated data
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Generated Data')
plt.show()

# Split data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
test_set_size = int(len(X) * TEST_SIZE)
test_indices = indices[:test_set_size]
train_indices = indices[test_set_size:]
X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Model parameters
n_features = X_train.shape[1]
n_output = 1

# Initialize weights and biases
W0 = np.zeros(n_output)
W1 = np.random.randn(n_output, n_features) * 0.1
# Forward and Backward Pass
def forward_pass(graph):
    for n in graph:
        n.forward()

def backward_pass(graph):
    for n in graph[::-1]:
        n.backward()


# SGD Update
def sgd_update(trainables, learning_rate=1e-2):
    for t in trainables:
        t.value -= learning_rate * t.gradients[t].T

batch_size= [1,2,4,8,10,12,24]
batch_loss_history = {}
for j in range(len(batch_size)):
    W0 = np.zeros(1)
    W = np.array(W1)
    # Create nodes
    x_node = Input()
    y_node = Input()
    b_node = Parameter(W0)
    a_node = Parameter(W)
    # Build computation graph
    u_node = Linear(x_node, a_node, b_node)
    sigmoid = Sigmoid(u_node)
    loss = BCE(y_node, sigmoid)
    # Create graph outside the training loop
    graph = [x_node, a_node, b_node, u_node, sigmoid, loss]
    trainable = [b_node, a_node]
    loss_per_epoch = []
    for epoch in range(epochs):
        loss_value = 0
        for i in range(0, X_train.shape[0], batch_size[j]):
            x_node.value = X_train[i:i + batch_size[j]]
            y_node.value = y_train[i:i + batch_size[j]].reshape(-1, 1)
            forward_pass(graph)
            backward_pass(graph)
            sgd_update(trainable, learning_rate)
            loss_value += loss.value
        avg_loss = loss_value / (X_train.shape[0] / batch_size[j])
        loss_per_epoch.append(avg_loss)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.6f} , Batch_size: {batch_size[j]}")
    batch_loss_history[batch_size[j]] = loss_per_epoch
# Evaluate the model
    correct_predictions = 0
    for i in range(0, X_test.shape[0], batch_size[j]):
        x_batch = X_test[i:i + batch_size[j]]
        y_batch = y_test[i:i + batch_size[j]].reshape(-1, 1)
        x_node.value = x_batch
        y_node.value = y_batch
        forward_pass(graph)
        for k in range(len(x_batch)):
            if (sigmoid.value[k] >= 0.5 and y_batch[k] == 1) or (sigmoid.value[k] < 0.5 and y_batch[k] == 0):
                correct_predictions += 1
    accuracy = correct_predictions / X_test.shape[0]
    print(f"Accuracy: {accuracy * 100:.02f}%")
#plot the batch size / loss
plt.figure(figsize=(10, 6))
for b_size, losses in batch_loss_history.items():
    plt.plot(range(1, len(losses) + 1), losses, label=f"Batch Size: {b_size}")
plt.xlabel("Epoch")
plt.ylabel("Average Training Loss")
plt.title("Effect of Batch Size on Training Loss")
plt.legend()
plt.grid()
plt.show()
x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 50),
    np.linspace(y_min, y_max, 50)
    )

Z = []
for i, j in zip(xx.ravel(), yy.ravel()):
    x_node.value = np.array([i, j]).reshape(1, -1)
    forward_pass(graph)
    if sigmoid.value.size == 1:
        Z.append(sigmoid.value.item())
    else:
        Z.append(sigmoid.value.mean())
Z = np.array(Z).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
plt.show()