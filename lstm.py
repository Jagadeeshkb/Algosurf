import numpy as np

class LSTM:
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize the LSTM network with weights and biases.

        Parameters:
        - input_dim: Number of input features.
        - hidden_dim: Number of hidden units in the LSTM cell.
        - output_dim: Number of output features.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize weights and biases for gates and candidate cell state
        self.Wf = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.01
        self.bf = np.zeros((hidden_dim, 1))

        self.Wi = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.01
        self.bi = np.zeros((hidden_dim, 1))

        self.Wo = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.01
        self.bo = np.zeros((hidden_dim, 1))

        self.Wc = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.01
        self.bc = np.zeros((hidden_dim, 1))

        # Initialize weights and biases for the output layer
        self.Wy = np.random.randn(output_dim, hidden_dim) * 0.01
        self.by = np.zeros((output_dim, 1))

    def sigmoid(self, x):
        """
        Sigmoid activation function.

        Parameters:
        - x: Input value(s).

        Returns:
        - Sigmoid activation value(s).
        """
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        """
        Tanh activation function.

        Parameters:
        - x: Input value(s).

        Returns:
        - Tanh activation value(s).
        """
        return np.tanh(x)

    def forward(self, x, h_prev, c_prev):
        """
        Forward pass through the LSTM cell.

        Parameters:
        - x: Input data (shape: (input_dim, 1)).
        - h_prev: Previous hidden state (shape: (hidden_dim, 1)).
        - c_prev: Previous cell state (shape: (hidden_dim, 1)).

        Returns:
        - h_next: Next hidden state.
        - c_next: Next cell state.
        - y: Output value.
        """
        # Concatenate input and previous hidden state
        concat = np.vstack((x, h_prev))

        # Compute forget gate
        self.f = self.sigmoid(np.dot(self.Wf, concat) + self.bf)
        # Compute input gate
        self.i = self.sigmoid(np.dot(self.Wi, concat) + self.bi)
        # Compute output gate
        self.o = self.sigmoid(np.dot(self.Wo, concat) + self.bo)
        # Compute candidate cell state
        self.c_tilde = self.tanh(np.dot(self.Wc, concat) + self.bc)
        
        # Compute next cell state
        self.c_next = self.f * c_prev + self.i * self.c_tilde
        # Compute next hidden state
        self.h_next = self.o * self.tanh(self.c_next)
        
        # Compute output
        self.y = np.dot(self.Wy, self.h_next) + self.by
        
        return self.h_next, self.c_next, self.y

    def backward(self, d_y, x, h_prev, c_prev, d_h_next, d_c_prev):
        """
        Backward pass through the LSTM cell to compute gradients.

        Parameters:
        - d_y: Gradient of the loss with respect to the output (shape: (output_dim, 1)).
        - x: Input data (shape: (input_dim, 1)).
        - h_prev: Previous hidden state (shape: (hidden_dim, 1)).
        - c_prev: Previous cell state (shape: (hidden_dim, 1)).
        - d_h_next: Gradient of the loss with respect to the next hidden state (shape: (hidden_dim, 1)).
        - d_c_prev: Gradient of the loss with respect to the previous cell state (shape: (hidden_dim, 1)).

        Returns:
        - d_x: Gradient of the loss with respect to the input data.
        - d_h_prev: Gradient of the loss with respect to the previous hidden state.
        - d_c_prev: Gradient of the loss with respect to the previous cell state.
        """
        # Gradient w.r.t. output layer
        self.d_Wy = np.dot(d_y, self.h_next.T)
        self.d_by = d_y
        
        # Gradient of the hidden state
        d_h = np.dot(self.Wy.T, d_y) + d_h_next
        
        # Gradient of the output gate
        d_o = d_h * self.tanh(self.c_next)
        d_o = d_o * self.sigmoid(self.o) * (1 - self.sigmoid(self.o))
        
        # Gradient of the cell state
        d_c = d_h * self.o * (1 - self.tanh(self.c_next)**2) + d_c_prev
        d_c_tilde = d_c * self.i
        d_c_tilde = d_c_tilde * (1 - self.tanh(self.c_tilde)**2)
        
        # Gradient of the input gate
        d_i = d_c * self.c_tilde
        d_i = d_i * self.sigmoid(self.i) * (1 - self.sigmoid(self.i))
        
        # Gradient of the forget gate
        d_f = d_c * c_prev
        d_f = d_f * self.sigmoid(self.f) * (1 - self.sigmoid(self.f))
        
        # Gradients w.r.t. concatenated input and previous hidden state
        concat = np.vstack((x, h_prev))
        d_concat = np.hstack((
            np.dot(self.Wf.T, d_f),
            np.dot(self.Wi.T, d_i),
            np.dot(self.Wo.T, d_o),
            np.dot(self.Wc.T, d_c_tilde)
        ))

        # Update gradients for weights and biases
        self.d_Wf = np.dot(d_f, concat.T)
        self.d_bf = np.sum(d_f, axis=1, keepdims=True)

        self.d_Wi = np.dot(d_i, concat.T)
        self.d_bi = np.sum(d_i, axis=1, keepdims=True)

        self.d_Wo = np.dot(d_o, concat.T)
        self.d_bo = np.sum(d_o, axis=1, keepdims=True)

        self.d_Wc = np.dot(d_c_tilde, concat.T)
        self.d_bc = np.sum(d_c_tilde, axis=1, keepdims=True)

        d_x = d_concat[:self.input_dim, :]
        d_h_prev = d_concat[self.input_dim:, :]

        return d_x, d_h_prev, d_c_prev

    def update_parameters(self, learning_rate):
        """
        Update parameters using the gradients obtained from the backward pass.

        Parameters:
        - learning_rate: Learning rate for parameter updates.
        """
        self.Wf -= learning_rate * self.d_Wf
        self.bf -= learning_rate * self.d_bf

        self.Wi -= learning_rate * self.d_Wi
        self.bi -= learning_rate * self.d_bi

        self.Wo -= learning_rate * self.d_Wo
        self.bo -= learning_rate * self.d_bo

        self.Wc -= learning_rate * self.d_Wc
        self.bc -= learning_rate * self.d_bc

        self.Wy -= learning_rate * self.d_Wy
        self.by -= learning_rate * self.d_by

# Example usage
if __name__ == "__main__":
    input_dim = 3    # Number of input features
    hidden_dim = 4   # Number of LSTM units
    output_dim = 2   # Number of output features

    lstm = LSTM(input_dim, hidden_dim, output_dim)

    # Dummy data
    x = np.random.randn(input_dim, 1)  # Input data (shape: (input_dim, 1))
    h_prev = np.zeros((hidden_dim, 1))  # Previous hidden state (shape: (hidden_dim, 1))
    c_prev = np.zeros((hidden_dim, 1))  # Previous cell state (shape: (hidden_dim, 1))

    # Forward pass
    h_next, c_next, y = lstm.forward(x, h_prev, c_prev)

    # Dummy gradients for the backward pass (for demonstration purposes)
    d_y = np.random.randn(output_dim, 1)
    d_h_next = np.zeros((hidden_dim, 1))
    d_c_prev = np.zeros((hidden_dim, 1))

    # Backward pass
    d_x, d_h_prev, d_c_prev = lstm.backward(d_y, x, h_prev, c_prev, d_h_next, d_c_prev)

    # Update parameters
    learning_rate = 0.001
    lstm.update_parameters(learning_rate)

    print("Next hidden state (h_next):", h_next)
    print("Next cell state (c_next):", c_next)
    print("Output (y):", y)
    print("Gradient with respect to input (d_x):", d_x)
    print("Gradient with respect to previous hidden state (d_h_prev):", d_h_prev)
    print("Gradient with respect to previous cell state (d_c_prev):", d_c_prev)
