#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 11:21:08 2024

@author: jaggu
"""

import numpy as np
from sklearn.model_selection import train_test_split

class SimpleRNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize the RNN with random weights and zero biases.

        Parameters:
        input_dim (int): Dimension of input features
        hidden_dim (int): Dimension of the hidden state
        output_dim (int): Dimension of the output
        """
        self.Wx = np.random.randn(input_dim, hidden_dim) * 0.01  # Weight matrix for input-to-hidden connections
        self.Wh = np.random.randn(hidden_dim, hidden_dim) * 0.01  # Weight matrix for hidden-to-hidden connections
        self.Why = np.random.randn(hidden_dim, output_dim) * 0.01  # Weight matrix for hidden-to-output connections
        self.bh = np.zeros((1, hidden_dim))  # Bias for hidden state
        self.by = np.zeros((1, output_dim))  # Bias for output

    def forward(self, X):
        """
        Forward pass through the RNN.

        Parameters:
        X (array): Input sequence of shape (batch_size, timesteps, input_dim)

        Returns:
        y (array): Output sequence of shape (batch_size, output_dim)
        """
        self.X = X
        batch_size, timesteps, _ = X.shape
        self.hidden_states = np.zeros((batch_size, timesteps, self.Wx.shape[1]))

        # Initial hidden state
        h = np.zeros((batch_size, self.Wx.shape[1]))

        # Forward pass through each timestep
        for t in range(timesteps):
            x_t = X[:, t, :]
            # Compute the hidden state using the previous hidden state and the current input
            h = np.tanh(np.dot(x_t, self.Wx) + np.dot(h, self.Wh) + self.bh)
            self.hidden_states[:, t, :] = h

        # Compute the output using the last hidden state
        y = np.dot(h, self.Why) + self.by
        return y

    def backward(self, dY, learning_rate=0.01):
        """
        Backward pass through the RNN to compute gradients and update weights.

        Parameters:
        dY (array): Gradient of the loss w.r.t. output of shape (batch_size, output_dim)
        learning_rate (float): Learning rate for weight updates
        """
        batch_size, _ = dY.shape
        dWhy = np.zeros_like(self.Why)
        dby = np.zeros_like(self.by)
        dWh = np.zeros_like(self.Wh)
        dbh = np.zeros_like(self.bh)
        dWx = np.zeros_like(self.Wx)

        # Gradient of the loss w.r.t. hidden state at the last timestep
        dH = np.zeros((batch_size, self.hidden_states.shape[2]))

        # Backward pass through each timestep in reverse
        for t in reversed(range(self.hidden_states.shape[1])):
            x_t = self.X[:, t, :]
            # Compute gradient of loss w.r.t. hidden state
            dh = dH + np.dot(dY, self.Why.T)
            # Compute gradient of tanh activation
            dtanh = (1 - self.hidden_states[:, t, :] ** 2) * dh
            dWhy += np.dot(self.hidden_states[:, t, :].T, dY)
            dby += np.sum(dY, axis=0, keepdims=True)
            dH = np.dot(dtanh, self.Wh.T)
            # Gradient w.r.t. hidden-to-hidden weight
            dWh += np.dot(self.hidden_states[:, t - 1, :].T, dtanh) if t > 0 else np.dot(x_t.T, dtanh)
            # Gradient w.r.t. input-to-hidden weight
            dWx += np.dot(x_t.T, dtanh)
            # Gradient w.r.t. hidden bias
            dbh += np.sum(dtanh, axis=0, keepdims=True)

        # Update weights and biases using gradient descent
        self.Wx -= learning_rate * dWx
        self.Wh -= learning_rate * dWh
        self.Why -= learning_rate * dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby

# Generate synthetic sequence data
def generate_sequence_data(seq_length, num_sequences):
    """
    Generate synthetic sequence data where each sequence consists of incremental numbers.

    Parameters:
    seq_length (int): Length of each sequence
    num_sequences (int): Number of sequences to generate

    Returns:
    X (array): Input sequences of shape (num_sequences, seq_length, 1)
    y (array): Target values of shape (num_sequences, 1)
    """
    X = np.zeros((num_sequences, seq_length, 1))
    y = np.zeros((num_sequences, 1))
    
    for i in range(num_sequences):
        start = np.random.randint(0, 10)
        seq = np.arange(start, start + seq_length).reshape(-1, 1)
        X[i] = seq
        y[i] = start + seq_length

    return X, y

# Generate data
seq_length = 5
num_sequences = 1000
X, y = generate_sequence_data(seq_length, num_sequences)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the RNN parameters
input_dim = 1  # Since each input is a single number
hidden_dim = 10
output_dim = 1  # Predicting the next number in the sequence

# Initialize and train the RNN
rnn = SimpleRNN(input_dim, hidden_dim, output_dim)

# Simple training loop (not optimized)
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    y_pred = rnn.forward(X_train)
    
    # Compute loss (mean squared error)
    loss = np.mean((y_pred - y_train) ** 2)
    
    # Compute gradient of the loss w.r.t. output
    dY = 2 * (y_pred - y_train) / y_train.shape[0]
    
    # Backward pass
    rnn.backward(dY, learning_rate=0.01)
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

# Evaluate on the test set
y_test_pred = rnn.forward(X_test)
test_loss = np.mean((y_test_pred - y_test) ** 2)
print(f'Test Loss: {test_loss:.4f}')
