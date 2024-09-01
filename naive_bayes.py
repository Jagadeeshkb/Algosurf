import numpy as np

class NaiveBayes:
    def fit(self, X, y):
        # Identify unique classes and their counts
        self.classes, class_counts = np.unique(y, return_counts=True)
        # Calculate prior probabilities for each class
        self.class_priors = class_counts / len(y)
        self.feature_probs = []
        
        # Calculate feature probabilities for each class
        for c in self.classes:
            X_c = X[y == c]  # Subset of X for class c
            feature_probs_c = []
            for feature in range(X.shape[1]):
                feature_values = np.unique(X[:, feature])
                feature_probs = {}
                for value in feature_values:
                    # Calculate probability of each feature value given the class
                    feature_probs[value] = np.sum(X_c[:, feature] == value) / len(X_c)
                feature_probs_c.append(feature_probs)
            self.feature_probs.append(feature_probs_c)
    
    def predict(self, X):
        predictions = []
        for sample in X:
            class_probs = []
            for i, c in enumerate(self.classes):
                prior = self.class_priors[i]
                # Calculate likelihood of the sample for class c
                likelihood = np.prod([self.feature_probs[i][f].get(value, 0) for f, value in enumerate(sample)])
                class_probs.append(prior * likelihood)
            # Predict the class with the highest probability
            predictions.append(self.classes[np.argmax(class_probs)])
        return np.array(predictions)


def generate_data(n_samples=100):
    # Initialize arrays
    X = np.zeros((n_samples, 2))
    y = np.zeros(n_samples, dtype=int)
    
    # Define the number of classes and pattern details
    n_classes = 3
    points_per_class = n_samples // n_classes
    
    # Generate the dataset
    index = 0
    for class_label in range(n_classes):
        for i in range(points_per_class):
            # Generate feature values following a pattern
            x1 = class_label + i % (n_classes + 1)
            x2 = (class_label + i) % (n_classes + 1)
            
            X[index] = [x1, x2]
            y[index] = class_label
            index += 1
    
    return X, y

# Generate the dataset
X, y = generate_data()

# Print the generated dataset
print("Feature matrix X:")
print(X)
print("\nClass labels y:")
print(y)

# Split dataset into training and test sets
X_train = X[:80]
y_train = y[:80]
X_test = X[80:]
y_test = y[80:]


# Train Naive Bayes classifier
nb = NaiveBayes()
nb.fit(X_train, y_train)

# Make predictions
y_pred = nb.predict(X_test)

# Print predictions and evaluate
print("Predictions:", y_pred)
print("True labels:", y_test)
