import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the Logistic Regression model.
        
        Parameters:
        - learning_rate: The learning rate for gradient descent.
        - num_iterations: The number of iterations for gradient descent.
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
    
    def sigmoid(self, z):
        """
        Compute the sigmoid function.
        
        The sigmoid function maps any real-valued number into the range [0, 1].
        
        Parameters:
        - z: Input value or array of values.
        
        Returns:
        - Sigmoid of the input value(s).
        """
        return 1 / (1 + np.exp(-z))
    
    def compute_cost(self, X, y):
        """
        Compute the cost function (binary cross-entropy loss).
        
        The cost function measures how well the model's predictions match the true labels.
        
        Parameters:
        - X: Feature matrix (with shape [m, n+1], where m is the number of samples and n is the number of features).
        - y: True labels (with shape [m,]).
        
        Returns:
        - Cost value as a scalar.
        """
        m = len(y)  # Number of training examples
        h = self.sigmoid(np.dot(X, self.theta))  # Hypothesis function
        cost = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return cost
    
    def gradient_descent(self, X, y):
        """
        Perform gradient descent to optimize the model parameters (theta).
        
        Parameters:
        - X: Feature matrix (with shape [m, n+1]).
        - y: True labels (with shape [m,]).
        """
        m = len(y)  # Number of training examples
        self.theta = np.zeros(X.shape[1])  # Initialize theta (weights) with zeros
        
        # Perform gradient descent
        for i in range(self.num_iterations):
            h = self.sigmoid(np.dot(X, self.theta))  # Predicted probabilities
            gradient = (1/m) * np.dot(X.T, (h - y))  # Compute gradient
            self.theta -= self.learning_rate * gradient  # Update theta
            
            # Print cost every 100 iterations for tracking
            if i % 100 == 0:
                cost = self.compute_cost(X, y)
                print(f"Iteration {i}: Cost {cost}")
    
    def fit(self, X, y):
        """
        Fit the logistic regression model to the training data.
        
        Parameters:
        - X: Feature matrix (with shape [m, n]).
        - y: True labels (with shape [m,]).
        """
        # Add intercept term (bias term) to feature matrix
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # Perform gradient descent to learn theta
        self.gradient_descent(X, y)
    
    def predict(self, X):
        """
        Predict class labels for the given feature matrix.
        
        Parameters:
        - X: Feature matrix (with shape [m, n]).
        
        Returns:
        - Predicted class labels (with shape [m,]).
        """
        # Add intercept term (bias term) to feature matrix
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # Compute predicted probabilities
        probs = self.sigmoid(np.dot(X, self.theta))
        
        # Classify as 1 if probability >= 0.5, else 0
        return np.where(probs >= 0.5, 1, 0)

# Example Usage
if __name__ == "__main__":
    # Example dataset with 5 samples and 2 features
    X = np.array([[1, 2],
                  [2, 3],
                  [3, 4],
                  [4, 5],
                  [5, 6]])
    y = np.array([0, 0, 1, 1, 1])  # Corresponding class labels
    
    # Initialize and train the logistic regression model
    model = LogisticRegression(learning_rate=0.1, num_iterations=1000)
    model.fit(X, y)
    
    # Make predictions on the training data
    predictions = model.predict(X)
    print("Predictions:", predictions)
