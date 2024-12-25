import numpy as np

# Base Node class
class Node:
    def __init__(self, inputs=None):
        if inputs is None:
            inputs = []
        self.inputs = inputs
        self.outputs = []
        self.value = None
        self.trainable=False
        self.visited=False
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
        self.trainable=True

    def forward(self):
        pass

    def backward(self):
        self.gradients = {self: 0}
        for n in self.outputs:
            self.gradients[self] += n.gradients[self]


class Linear(Node):
    def __init__(self, x,output_size,input_size , initialization_controller=0.1):
        W0 = np.zeros(output_size)
        W1 = np.random.randn(output_size, input_size) * initialization_controller
        a = Parameter(W1)
        b = Parameter(W0)
        Node.__init__(self, [x, a, b])

    def forward(self):
        x, a, b = self.inputs
        self.value = np.dot(x.value, a.value.T) + b.value

    def backward(self):
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
class ReLU(Node):
    def __init__(self, x):
        Node.__init__(self, [x])

    def forward(self):
        x = self.inputs[0]
        self.value = np.maximum(0, x.value)

    def backward(self):
        x = self.inputs[0]
        #if self not in self.gradients:
            #self.gradients[self] = 0
        self.gradients[x] = self.outputs[0].gradients[self] * (x.value > 0).astype(float)

class BCE(Node):
    def __init__(self, y_true, y_pred):
        Node.__init__(self, [y_true, y_pred])

    def forward(self):
        y_true, y_pred = self.inputs
        epsilon = 1e-12
        y_pred = np.clip(y_pred.value, epsilon, 1 - epsilon)
        self.value = -np.mean(
            y_true.value * np.log(y_pred) + (1 - y_true.value) * np.log(1 - y_pred)
        )

    def backward(self):
        y_true, y_pred, = self.inputs
        batch_size = y_true.value.shape[0]
        epsilon = 1e-12
        y_pred = np.clip(y_pred.value, epsilon, 1 - epsilon)
        self.gradients[self.inputs[1]] = (1 / batch_size) * (y_pred - y_true.value) / (y_pred * (1 - y_pred))
        self.gradients[y_true] = (1 / batch_size) * (np.log(y_pred) - np.log(1 - y_pred))


class Softmax(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def forward(self):
        input_value = self.inputs[0].value
        exp_values = np.exp(input_value - np.max(input_value, axis=1, keepdims=True))  # Stability
        self.value = exp_values / np.sum(exp_values, axis=1, keepdims=True)  # Row-wise normalization

    def backward(self):
        grad_output = self.outputs[0].gradients[self]  # Gradients passed from next layer
        self.gradients[self.inputs[0]] = grad_output
class CE(Node):
    def __init__(self, y_true, y_pred):
        Node.__init__(self, [y_true, y_pred])
    def forward(self):
        y_true = self.inputs[0].value
        y_pred = self.inputs[1].value
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        self.value = -np.mean(np.log(y_pred[np.arange(len(y_true)), y_true]))
    def backward(self):
        y_true = self.inputs[0].value
        y_pred = self.inputs[1].value
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        grad_y_pred = y_pred.copy()
        grad_y_pred[np.arange(len(y_true)), y_true] -= 1
        grad_y_pred /= len(y_true)
        self.gradients[self.inputs[0]] = np.zeros_like(y_true)
        self.gradients[self.inputs[1]] = grad_y_pred
class Conv:
    def __init__(self, node, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        Node.__init__(self, [node])
        self.trainable = True
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self._w = np.random.normal(0, 0.1, (kernel_size, kernel_size, in_channels, out_channels))
        self._b = np.ones((1, 1, 1, out_channels))  # Bias initialized for each filter

    def calculate_output_dims(self, input_dims: tuple) -> tuple:
        n, h_in, w_in, c_in = input_dims
        h_f, w_f, c_f, n_f = self._w.shape
        stride = self.stride

        h_out = (h_in - h_f + 2 * self.padding) // stride + 1
        w_out = (w_in - w_f + 2 * self.padding) // stride + 1
        return (n, h_out, w_out, n_f)

    def pad_input(self, a_prev):
        if self.padding == 0:
            return a_prev
        return np.pad(a_prev,
                      ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                      mode='constant', constant_values=0)

    def forward(self):
        a_prev = self.inputs[0].value
        a_prev_padded = self.pad_input(a_prev)
        output_shape = self.calculate_output_dims(a_prev.shape)
        n, h_in, w_in, _ = a_prev_padded.shape
        _, h_out, w_out, _ = output_shape
        h_f, w_f, _, n_f = self._w.shape

        output = np.zeros(output_shape)

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = h_start + h_f
                w_end = w_start + w_f

                a_slice = a_prev_padded[:, h_start:h_end, w_start:w_end, :]
                output[:, i, j, :] = np.sum(a_slice[:, :, :, :, np.newaxis] * self._w[np.newaxis, :, :, :], axis=(1, 2, 3))

        self.value = output + self._b

    def backward(self):
        grad_output = self.outputs[0].gradients[self]
        a_prev = self.inputs[0].value
        a_prev_padded = self.pad_input(a_prev)
        n, h_in, w_in, c_in = a_prev.shape
        h_f, w_f, _, n_f = self._w.shape
        _, h_out, w_out, _ = grad_output.shape

        d_w = np.zeros_like(self._w)
        d_b = np.sum(grad_output, axis=(0, 1, 2))
        d_a_prev_padded = np.zeros_like(a_prev_padded)

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = h_start + h_f
                w_end = w_start + w_f

                a_slice = a_prev_padded[:, h_start:h_end, w_start:w_end, :]

                d_w += np.sum(a_slice[:, :, :, :, np.newaxis] * grad_output[:, i:i+1, j:j+1, np.newaxis, :], axis=0)
                d_a_prev_padded[:, h_start:h_end, w_start:w_end, :] += np.sum(
                    self._w[np.newaxis, :, :, :, :] * grad_output[:, i:i+1, j:j+1, np.newaxis, :],
                    axis=4
                )

        if self.padding > 0:
            d_a_prev = d_a_prev_padded[:, self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            d_a_prev = d_a_prev_padded

        self.gradients['weights'] = d_w
        self.gradients['bias'] = d_b
        self.gradients[self.inputs[0]] = d_a_prev

class MaxPooling:
    def __init__(self, node, kernel_size=2, stride=2):
        Node.__init__(self, [node])
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self):
        a_prev = self.inputs[0].value
        self._shape = a_prev.shape
        n, h_in, w_in, c = a_prev.shape
        h_pool, w_pool = self.kernel_size, self.kernel_size
        h_out = 1 + (h_in - h_pool) // self.stride
        w_out = 1 + (w_in - w_pool) // self.stride
        output = np.zeros((n, h_out, w_out, c))

        self._mask = {}

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = h_start + h_pool
                w_end = w_start + w_pool
                a_slice = a_prev[:, h_start:h_end, w_start:w_end, :]
                self._mask[(i, j)] = (a_slice == np.max(a_slice, axis=(1, 2), keepdims=True))
                output[:, i, j, :] = np.max(a_slice, axis=(1, 2))

        self.value = output

    def backward(self):
        grad_output = self.outputs[0].gradients[self]
        n, h_in, w_in, c = self._shape
        h_pool, w_pool = self.kernel_size, self.kernel_size
        d_a_prev = np.zeros(self._shape)

        h_out, w_out, _ = grad_output.shape[1:]

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = h_start + h_pool
                w_end = w_start + w_pool

                mask = self._mask[(i, j)]
                d_a_prev[:, h_start:h_end, w_start:w_end, :] += mask * grad_output[:, i:i+1, j:j+1, :]

        self.gradients[self.inputs[0]] = d_a_prev

class Flatten(Node):
    def __init__(self, input_node):
        Node.__init__(self, [input_node])

    def forward(self):
        input_value = self.inputs[0].value
        self.value = input_value.reshape(input_value.shape[0], -1)

    def backward(self):
        grad_output = self.outputs[0].gradients[self]
        input_shape = self.inputs[0].value.shape
        self.gradients[self.inputs[0]] = grad_output.reshape(input_shape)