import numpy as np

# Base Node class
class Node:
    def __init__(self, inputs=None):
        if inputs is None:
            inputs = []
        self.inputs = inputs
        self.outputs = []
        self.value = None
        self.gradients = {}
        for node in inputs:
            node.outputs.append(self)

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


# Input Node
class Input(Node):
    def __init__(self):
        Node.__init__(self)

    def forward(self, value=None):
        if value is not None:
            self.value = value

    def backward(self):
        self.gradients = {self: 0}
        for n in self.outputs:
            self.gradients[self] += n.gradients[self]


# Parameter Node
class Parameter(Node):
    def __init__(self, value):
        Node.__init__(self)
        self.value = value

    def forward(self):
        pass

    def backward(self):
        self.gradients = {self: 0}
        for n in self.outputs:
            self.gradients[self] += n.gradients[self]
class Linear(Node):
    def __init__(self,x,a,b):
        Node.__init__(self,[x,a,b])
    def forward(self):
        x,a,b=self.inputs
        self.value = np.dot(x.value , a.value.T) + b.value
    def backward(self):
        x,a,b=self.inputs
        x, a, b = self.inputs
        self.gradients[x] = np.dot(self.outputs[0].gradients[self], a.value)
        self.gradients[a] = np.dot(x.value.T, self.outputs[0].gradients[self])
        self.gradients[b] = np.sum(self.outputs[0].gradients[self], axis=0)

# Sigmoid Activation Node
class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self):
        input_value = self.inputs[0].value
        self.value = self._sigmoid(input_value)

    def backward(self):
        partial = self.value * (1 - self.value)
        self.gradients[self.inputs[0]] = partial * self.outputs[0].gradients[self]

class BCE(Node):
    def __init__(self, y_true, y_pred):
        Node.__init__(self, [y_true, y_pred])

    def forward(self):
        y_true, y_pred = self.inputs
        epsilon = 1e-12
        y_pred = np.clip(y_pred.value, epsilon, 1 - epsilon)
        self.value = -np.mean(
            y_true.value * np.log(y_pred) +(1 - y_true.value) * np.log(1 - y_pred)
        )

    def backward(self):
        y_true , y_pred,= self.inputs
        batch_size = y_true.value.shape[0]
        epsilon = 1e-12
        y_pred = np.clip(y_pred.value, epsilon, 1 - epsilon)
        self.gradients[self.inputs[1]] = (1 / batch_size) * (y_pred - y_true.value) / (y_pred * (1 - y_pred))
        self.gradients[y_true] = (1 / batch_size) * (np.log(y_pred) - np.log(1 - y_pred))