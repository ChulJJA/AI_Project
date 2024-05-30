#AI Project: Digit and Face Classification

##Overview

This project implements classification methods for digit and face recognition using various algorithms, including Perceptron and Neural Network classifiers. The project utilizes training data to classify digit images from the MNIST dataset and face images from the Caltech 101 dataset.

##Repository Structure

classificationMethod.py: Defines the abstract base class for classification methods.

dataClassifier.py: Contains feature extraction methods and harness code for data classification.

neuralNetwork.py: Implements the Neural Network classifier.

perceptron.py: Implements the Perceptron classifier.

samples.py: Provides utility functions for loading and processing image data.

util.py: Contains utility functions and data structures used across the project.


**Run the classifier:**

To execute the data classifier, use the following command:

python dataClassifier.py -c [classifier] -d [dataset] -t [training_size] -i [iterations] -s [test_size]

classifier: Type of classifier (perceptron or neural).

dataset: Dataset to use (digits or faces).

training_size: Size of the training set.

iterations: Maximum iterations for training.

test_size: Amount of test data to use.

Example:

python dataClassifier.py -c perceptron -d digits -t 1000 -i 10 -s 100

