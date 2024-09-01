import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        # Initialize the SVM with hyperparameters
        self.learning_rate = learning_rate  # Learning rate for gradient descent
        self.lambda_param = lambda_param    # Regularization parameter
        self.n_iters = n_iters              # Number of iterations for training
        self.w = None                       # Weights (to be learned)
        self.b = None                       # Bias (to be learned)

    def fit(self, X, y):
        # Fit the SVM model to the training data
        n_samples, n_features = X.shape     # Number of samples and features
        y_ = np.where(y <= 0, -1, 1)        # Convert labels to -1 and 1
        
        self.w = np.zeros(n_features)       # Initialize weights to zeros
        self.b = 0                          # Initialize bias to zero

        for _ in range(self.n_iters):       # Iterate over the number of iterations
            for idx, x_i in enumerate(X):   # Iterate over each sample
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    # If the condition is met, update weights with regularization term
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    # If the condition is not met, update weights and bias
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]

    def predict(self, X):
        # Predict the class labels for the input data
        approx = np.dot(X, self.w) - self.b  # Calculate the linear combination
        return np.sign(approx)               # Return the sign of the result as the prediction

# Example usage:
if __name__ == "__main__":
    # Sample data
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [1, 0], [0, 1], [2, 1], [3, 0]])
    y = np.array([1, 1, 1, 1, -1, -1, -1, -1])

    # Train SVM
    clf = SVM()                             # Create an instance of the SVM class
    clf.fit(X, y)                           # Fit the model to the sample data
    predictions = clf.predict(X)            # Predict the labels for the sample data

    print("Predictions:", predictions)      # Print the predictions
