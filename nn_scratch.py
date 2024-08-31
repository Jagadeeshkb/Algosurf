#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 08:26:04 2024

@author: jaggu
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Feature matrix (4 features per sample)
y = iris.target  # Target vector (class labels for each sample)

# Normalize the feature matrix to have zero mean and unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Apply scaling to the features

# Convert the target vector into one-hot encoded format
encoder = OneHotEncoder()
y_one_hot = encoder.fit_transform(y.reshape(-1, 1)).toarray()  # Reshape y and apply one-hot encoding

# Split the dataset into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_one_hot, test_size=0.2, random_state=42)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the neural network with weights and biases.
        
        Parameters:
        input_size (int): Number of input features
        hidden_size (int): Number of neurons in the hidden layer
        output_size (int): Number of output classes (number of neurons in the output layer)
        """
        # Initialize weights and biases for the hidden and output layers
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01  # Weights for input to hidden layer
        self.b1 = np.zeros((1, hidden_size))  # Biases for hidden layer
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01  # Weights for hidden to output layer
        self.b2 = np.zeros((1, output_size))  # Biases for output layer
    
    def relu(self, z):
        """
        Apply the ReLU activation function.
        
        Parameters:
        z (array): Input array to the activation function
        
        Returns:
        array: Result of applying ReLU activation function
        """
        return np.maximum(0, z)
    
    def softmax(self, z):
        """
        Apply the softmax activation function.
        
        Parameters:
        z (array): Input array to the activation function
        
        Returns:
        array: Result of applying softmax activation function
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Numerical stability improvement
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        Perform forward propagation through the network.
        
        Parameters:
        X (array): Input features
        
        Returns:
        array: Predicted probabilities for each class
        """
        self.z1 = np.dot(X, self.W1) + self.b1  # Linear combination for hidden layer
        self.a1 = self.relu(self.z1)  # Apply ReLU activation
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # Linear combination for output layer
        self.a2 = self.softmax(self.z2)  # Apply softmax activation
        return self.a2
    
    def compute_loss(self, y_true, y_pred):
        """
        Compute the cross-entropy loss between the true and predicted values.
        
        Parameters:
        y_true (array): True one-hot encoded labels
        y_pred (array): Predicted probabilities
        
        Returns:
        float: Loss value
        """
        m = y_true.shape[0]  # Number of samples
        log_likelihood = -np.log(y_pred[np.arange(m), np.argmax(y_true, axis=1)])  # Compute log likelihood
        return np.sum(log_likelihood) / m  # Average loss
    
    def backward(self, X, y_true, y_pred, learning_rate=0.01):
        """
        Perform backward propagation to update the weights and biases.
        
        Parameters:
        X (array): Input features
        y_true (array): True one-hot encoded labels
        y_pred (array): Predicted probabilities
        learning_rate (float): Learning rate for weight updates
        """
        m = y_true.shape[0]  # Number of samples
        
        # Compute gradients for the output layer
        dz2 = y_pred - y_true  # Gradient of loss w.r.t. z2
        dW2 = np.dot(self.a1.T, dz2) / m  # Gradient of loss w.r.t. W2
        db2 = np.sum(dz2, axis=0) / m  # Gradient of loss w.r.t. b2
        
        # Compute gradients for the hidden layer
        dz1 = np.dot(dz2, self.W2.T) * (self.a1 > 0)  # Gradient of loss w.r.t. z1 (ReLU derivative)
        dW1 = np.dot(X.T, dz1) / m  # Gradient of loss w.r.t. W1
        db1 = np.sum(dz1, axis=0) / m  # Gradient of loss w.r.t. b1
        
        # Update weights and biases
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
    
    def train(self, X_train, y_train, epochs=1000, learning_rate=0.01):
        """
        Train the neural network using gradient descent.
        
        Parameters:
        X_train (array): Training features
        y_train (array): Training labels
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for weight updates
        """
        for epoch in range(epochs):
            y_pred = self.forward(X_train)  # Forward pass
            loss = self.compute_loss(y_train, y_pred)  # Compute loss
            self.backward(X_train, y_train, y_pred, learning_rate)  # Backward pass
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')  # Print loss every 100 epochs
    
    def predict(self, X):
        """
        Predict the class labels for given input features.
        
        Parameters:
        X (array): Input features
        
        Returns:
        array: Predicted class labels
        """
        y_pred = self.forward(X)  # Forward pass
        return np.argmax(y_pred, axis=1)  # Return index of highest probability class

# Initialize the neural network with input size, hidden layer size, and output size
input_size = X_train.shape[1]  # Number of features
hidden_size = 10  # Number of neurons in hidden layer
output_size = y_train.shape[1]  # Number of classes
nn = NeuralNetwork(input_size, hidden_size, output_size)  # Create neural network instance

# Train the neural network
nn.train(X_train, y_train, epochs=1001, learning_rate=0.01)

# Predict on the test set and compute accuracy
y_test_pred = nn.predict(X_test)  # Predict labels for the test set
y_test_true = np.argmax(y_test, axis=1)  # Convert one-hot encoded test labels to single class labels

# Calculate accuracy
accuracy = np.mean(y_test_pred == y_test_true)  # Compute accuracy
print(f'Test Accuracy: {accuracy * 100:.2f}%')  # Print test accuracy
