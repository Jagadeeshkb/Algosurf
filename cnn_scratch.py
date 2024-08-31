import numpy as np

class Conv2D:
    def __init__(self, num_filters, filter_size):
        """
        Initialize the convolutional layer with random weights.

        Parameters:
        num_filters (int): Number of filters (or kernels)
        filter_size (int): Size of the square filter (filter_size x filter_size)
        """
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size) * 0.01  # Random weights
        self.biases = np.zeros(num_filters)  # Biases for each filter

    def forward(self, X):
        """
        Forward pass through the convolutional layer.

        Parameters:
        X (array): Input data of shape (batch_size, height, width, channels)

        Returns:
        out (array): Output data after convolution
        """
        self.X = X
        batch_size, height, width, channels = X.shape
        out_height = height - self.filter_size + 1
        out_width = width - self.filter_size + 1

        self.out = np.zeros((batch_size, out_height, out_width, self.num_filters))

        # Apply each filter to the input
        for i in range(self.num_filters):
            filter = self.filters[i]
            bias = self.biases[i]
            for b in range(batch_size):
                for h in range(out_height):
                    for w in range(out_width):
                        self.out[b, h, w, i] = np.sum(X[b, h:h+self.filter_size, w:w+self.filter_size] * filter) + bias

        return self.out

    def backward(self, d_out, learning_rate=0.01):
        """
        Backward pass through the convolutional layer to compute gradients.

        Parameters:
        d_out (array): Gradient of the loss w.r.t. the output
        learning_rate (float): Learning rate for weight updates
        """
        batch_size, out_height, out_width, _ = d_out.shape
        d_filters = np.zeros_like(self.filters)
        d_biases = np.zeros_like(self.biases)
        d_X = np.zeros_like(self.X)

        for i in range(self.num_filters):
            filter = self.filters[i]
            d_filter = np.zeros_like(filter)
            d_bias = 0
            for b in range(batch_size):
                for h in range(out_height):
                    for w in range(out_width):
                        d_filter += d_out[b, h, w, i] * self.X[b, h:h+self.filter_size, w:w+self.filter_size]
                        d_bias += d_out[b, h, w, i]
                        d_X[b, h:h+self.filter_size, w:w+self.filter_size] += d_out[b, h, w, i] * filter

            d_filters[i] = d_filter
            d_biases[i] = d_bias

        self.filters -= learning_rate * d_filters
        self.biases -= learning_rate * d_biases

        return d_X

class ReLU:
    def forward(self, X):
        """
        Forward pass through the ReLU activation function.

        Parameters:
        X (array): Input data

        Returns:
        out (array): Output data after ReLU activation
        """
        self.X = X
        return np.maximum(0, X)

    def backward(self, d_out):
        """
        Backward pass through the ReLU activation function.

        Parameters:
        d_out (array): Gradient of the loss w.r.t. the output

        Returns:
        d_X (array): Gradient of the loss w.r.t. the input
        """
        return d_out * (self.X > 0)

class MaxPooling:
    def __init__(self, pool_size):
        """
        Initialize the max pooling layer.

        Parameters:
        pool_size (int): Size of the square pooling window
        """
        self.pool_size = pool_size

    def forward(self, X):
        """
        Forward pass through the max pooling layer.

        Parameters:
        X (array): Input data of shape (batch_size, height, width, channels)

        Returns:
        out (array): Output data after max pooling
        """
        self.X = X
        batch_size, height, width, channels = X.shape
        out_height = height // self.pool_size
        out_width = width // self.pool_size

        self.out = np.zeros((batch_size, out_height, out_width, channels))

        for b in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    self.out[b, h, w, :] = np.max(X[b, h*self.pool_size:(h+1)*self.pool_size, w*self.pool_size:(w+1)*self.pool_size, :], axis=(0, 1))

        return self.out

    def backward(self, d_out):
        """
        Backward pass through the max pooling layer to compute gradients.

        Parameters:
        d_out (array): Gradient of the loss w.r.t. the output

        Returns:
        d_X (array): Gradient of the loss w.r.t. the input
        """
        batch_size, out_height, out_width, channels = d_out.shape
        d_X = np.zeros_like(self.X)

        for b in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    pool_region = self.X[b, h*self.pool_size:(h+1)*self.pool_size, w*self.pool_size:(w+1)*self.pool_size, :]
                    max_mask = (pool_region == np.max(pool_region, axis=(0, 1), keepdims=True))
                    d_X[b, h*self.pool_size:(h+1)*self.pool_size, w*self.pool_size:(w+1)*self.pool_size, :] += d_out[b, h, w, :] * max_mask

        return d_X

class FullyConnected:
    def __init__(self, input_dim, output_dim):
        """
        Initialize the fully connected layer with random weights.

        Parameters:
        input_dim (int): Number of input features
        output_dim (int): Number of output features
        """
        self.weights = np.random.randn(input_dim, output_dim) * 0.01  # Random weights
        self.biases = np.zeros(output_dim)  # Biases

    def forward(self, X):
        """
        Forward pass through the fully connected layer.

        Parameters:
        X (array): Input data of shape (batch_size, input_dim)

        Returns:
        out (array): Output data after the fully connected layer
        """
        self.X = X
        return np.dot(X, self.weights) + self.biases

    def backward(self, d_out, learning_rate=0.01):
        """
        Backward pass through the fully connected layer to compute gradients.

        Parameters:
        d_out (array): Gradient of the loss w.r.t. the output
        learning_rate (float): Learning rate for weight updates

        Returns:
        d_X (array): Gradient of the loss w.r.t. the input
        """
        d_weights = np.dot(self.X.T, d_out)
        d_biases = np.sum(d_out, axis=0)
        d_X = np.dot(d_out, self.weights.T)

        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases

        return d_X

# Example Usage
if __name__ == "__main__":
    # Hyperparameters
    input_height, input_width = 28, 28  # MNIST image size
    num_filters = 8
    filter_size = 3
    pool_size = 2
    num_fc_units = 10  # Number of classes in MNIST (digits 0-9)
    learning_rate = 0.01

    # Create layers
    conv = Conv2D(num_filters, filter_size)
    relu = ReLU()
    pool = MaxPooling(pool_size)
    fc = FullyConnected((input_height // pool_size) * (input_width // pool_size) * num_filters, num_fc_units)

    # Create a synthetic dataset (batch_size, height, width, channels)
    X = np.random.randn(10, input_height, input_width, 1)  # Batch of 10 images
    y = np.random.randint(0, num_fc_units, size=(10,))  # Random labels

    # Forward pass
    out = conv.forward(X)
    out = relu.forward(out)
    out = pool.forward(out)
    out = out.reshape(out.shape[0], -1)  # Flatten
    out = fc.forward(out)

    print("Output shape after fully connected layer:", out.shape)

    # Assuming a dummy loss gradient for demonstration
    d_out = np.random.randn(*out.shape)
    d_out = fc.backward(d_out, learning_rate)
    d_out = d_out.reshape(X.shape[0], input_height // pool_size, input_width // pool_size, num_filters)
    d_out = pool.backward(d_out)
    d_out = relu.backward(d_out)
    d_out = conv.backward(d_out, learning_rate)
