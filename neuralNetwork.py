import numpy as np
import scipy.optimize as opt

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class NeuralNetworkClassifier():
    def __init__(self, legalLabels,inputNum, hiddenNum, outputNum, trainingData, lambda_):
        self.input = inputNum
        self.hidden = hiddenNum
        self.output = outputNum
        self.traningData = trainingData
        self.legalLabels = legalLabels
        self.lambda_ = lambda_

        self.inputActivation = np.ones((self.input + 1, trainingData)) 
        self.hiddenActivation = np.ones((self.hidden + 1, trainingData)) 
        self.outputActivation = np.ones((self.output, trainingData))

        self.inputChange = np.zeros((self.hidden, self.input + 1))
        self.outputChange = np.zeros((self.output, self.hidden + 1))

        self.hiddenEpsilon = np.sqrt(6.0 / (self.input + self.hidden))
        self.outputEpsilon = np.sqrt(6.0 / (self.input + self.output))
        self.inputWeights = np.random.rand(self.hidden, self.input + 1) * 2 * self.hiddenEpsilon - self.hiddenEpsilon
        self.outputWeights = np.random.rand(self.output, self.hidden + 1) * 2 * self.outputEpsilon - self.outputEpsilon

    def feedForward(self, thetaVec):
        self.inputWeights = thetaVec[0:self.hidden * (self.input + 1)].reshape((self.hidden, self.input + 1))
        self.outputWeights = thetaVec[-self.output * (self.hidden + 1):].reshape((self.output, self.hidden + 1))

        self.hiddenActivation[:-1, :] = sigmoid(self.inputWeights.dot(self.inputActivation))
        self.outputActivation = sigmoid(self.outputWeights.dot(self.hiddenActivation))

        costMatrix = self.outputTruth * np.log(self.outputActivation) + (1 - self.outputTruth) * np.log(1 - self.outputActivation)
        regulations = (np.sum(self.outputWeights[:, :-1] ** 2) + np.sum(self.inputWeights[:, :-1] ** 2)) * self.lambda_ / 2
        return (-costMatrix.sum() + regulations) / self.traningData

    def backPropagate(self, thetaVec):
        self.inputWeights = thetaVec[0:self.hidden * (self.input + 1)].reshape((self.hidden, self.input + 1))
        self.outputWeights = thetaVec[-self.output * (self.hidden + 1):].reshape((self.output, self.hidden + 1))

        outputError = self.outputActivation - self.outputTruth
        hiddenError = self.outputWeights[:, :-1].T.dot(outputError) * (self.hiddenActivation[:-1:] * (1.0 - self.hiddenActivation[:-1:]))

        self.outputChange = outputError.dot(self.hiddenActivation.T) / self.traningData
        self.inputChange = hiddenError.dot(self.inputActivation.T) / self.traningData

        self.outputChange[:, :-1].__add__(self.lambda_ * self.outputWeights[:, :-1])
        self.inputChange[:, :-1].__add__(self.lambda_ * self.inputWeights[:, :-1])

        return np.append(self.inputChange.ravel(), self.outputChange.ravel())

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        self.size_train = len(list(trainingData))
        features_train = [];
        for datum in trainingData:
            feature = list(datum.values())
            features_train.append(feature)
        train_set = np.array(features_train, np.int32)

        iteration = 100
        self.inputActivation[:-1, :] = train_set.transpose()
        self.outputTruth = self.truthMatrix(trainingLabels)
        
        self.accuracies = []  # List to store accuracies

        def callback(thetaVec):
            self.feedForward(thetaVec) 
            accuracy = self.calculate_accuracy(self.outputActivation, self.outputTruth)
            self.accuracies.append(accuracy)

        thetaVec = np.append(self.inputWeights.ravel(), self.outputWeights.ravel())
        thetaVec = opt.fmin_cg(self.feedForward, thetaVec, fprime=self.backPropagate, maxiter=iteration, callback=callback)
        self.inputWeights = thetaVec[0:self.hidden * (self.input + 1)].reshape((self.hidden, self.input + 1))
        self.outputWeights = thetaVec[-self.output * (self.hidden + 1):].reshape((self.output, self.hidden + 1))
        
        return self.accuracies
        
    def calculate_accuracy(self, outputActivations, labels):
        predictions = np.argmax(outputActivations, axis=0) 
        if labels.shape[0] == 1: 
            correct_predictions = np.sum((outputActivations > 0.5) == labels) 
        else:  
            true_labels = np.argmax(labels, axis=0)  
            correct_predictions = np.sum(predictions == true_labels)
        accuracy = correct_predictions / labels.shape[1] 
        return accuracy


    def classify(self, data):
        self.size_test = len(list(data))
        features_test = [];
        for datum in data:
            feature = list(datum.values())
            features_test.append(feature)
        test_set = np.array(features_test, np.int32)
        feature_test_set = test_set.transpose()

        if feature_test_set.shape[1] != self.inputActivation.shape[1]:
            self.inputActivation = np.ones((self.input + 1, feature_test_set.shape[1]))
            self.hiddenActivation = np.ones((self.hidden + 1, feature_test_set.shape[1]))
            self.outputActivation = np.ones((self.output + 1, feature_test_set.shape[1]))
        self.inputActivation[:-1, :] = feature_test_set

        self.hiddenActivation[:-1, :] = sigmoid(self.inputWeights.dot(self.inputActivation))

        self.outputActivation = sigmoid(self.outputWeights.dot(self.hiddenActivation))
        if self.output > 1:
            return np.argmax(self.outputActivation, axis=0).tolist()
        else:
            return (self.outputActivation>0.5).ravel()

    def truthMatrix(self, trainLabels):
        truth = np.zeros((self.output, self.traningData))
        for i in range(self.traningData):
            label = trainLabels[i]
            if self.output == 1:
                truth[:,i] = label
            else:
                truth[label, i] = 1
        return truth
