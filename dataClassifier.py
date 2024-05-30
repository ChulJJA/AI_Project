# dataClassifier.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# This file contains feature extraction methods and harness 
# code for data classification

import samples
import sys
import util
import perceptron
import neuralNetwork
import time
import numpy as np
import random

TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70

def basicFeatureExtractorDigit(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is white (0) or gray/black (1)
  """
  a = datum.getPixels()

  features = util.Counter()
  for x in range(DIGIT_DATUM_WIDTH):
    for y in range(DIGIT_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

def basicFeatureExtractorFace(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is an edge (1) or no edge (0)
  """
  a = datum.getPixels()

  features = util.Counter()
  for x in range(FACE_DATUM_WIDTH):
    for y in range(FACE_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
  """
  This function is called after learning.
  Include any code that you want here to help you analyze your results.
  
  Use the printImage(<list of pixels>) function to visualize features.
  
  An example of use has been given to you.
  
  - classifier is the trained classifier
  - guesses is the list of labels predicted by your classifier on the test set
  - testLabels is the list of true labels
  - testData is the list of training datapoints (as util.Counter of features)
  - rawTestData is the list of training datapoints (as samples.Datum)
  - printImage is a method to visualize the features 
  (see its use in the odds ratio part in runClassifier method)
  
  This code won't be evaluated. It is for your own optional use
  (and you can modify the signature if you want).
  """
  
  # Put any code here...
  # Example of use:
  for i in range(len(guesses)):
      prediction = guesses[i]
      truth = testLabels[i]
      if (prediction != truth):
          print("===================================")
          print("Mistake on example {}".format(i))
          print("Predicted {}; truth is {}".format(prediction, truth))
          print("Image: ")
          print(rawTestData[i])
          break

## =====================
## You don't have to modify any code below.
## =====================


class ImagePrinter:
    def __init__(self, width, height):
      self.width = width
      self.height = height

    def printImage(self, pixels):
      """
      Prints a Datum object that contains all pixels in the 
      provided list of pixels.  This will serve as a helper function
      to the analysis function you write.
      
      Pixels should take the form 
      [(2,2), (2, 3), ...] 
      where each tuple represents a pixel.
      """
      image = samples.Datum(None,self.width,self.height)
      for pix in pixels:
        try:
            # This is so that new features that you could define which 
            # which are not of the form of (x,y) will not break
            # this image printer...
            x,y = pix
            image.pixels[x][y] = 2
        except:
            print("new features:", pix)
            continue
      print(image) 

def default(str):
  return str + ' [Default: %default]'

def readCommand( argv ):
  "Processes the command used to run from the command line."
  from optparse import OptionParser  
  parser = OptionParser(USAGE_STRING)
  
  parser.add_option('-c', '--classifier', help=default('The type of classifier'), choices=['perceptron', 'neural'], default='perceptron')
  parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'faces'], default='faces')
  parser.add_option('-t', '--training', help=default('The size of the training set'), default=150, type="int")
  parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=3, type="int")
  parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")

  options, otherjunk = parser.parse_args(argv)
  if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
  args = {}
  
  # Set up variables according to the command line input.
  print("Doing classification")
  print("--------------------")
  print("data:\t\t{}".format(options.data))
  print("classifier:\t\t{}".format(options.classifier))
  print("training set size:\t{}".format(str(options.training)))
  if(options.data=="digits"):
    printImage = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).printImage
    featureFunction = basicFeatureExtractorDigit
  elif(options.data=="faces"):
    printImage = ImagePrinter(FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT).printImage
    featureFunction = basicFeatureExtractorFace      
  else:
    print("Unknown dataset", options.data)
    print(USAGE_STRING)
    sys.exit(2)
    
  if(options.data=="digits"):
    legalLabels = range(10)
  else:
    legalLabels = range(2)

  if(options.classifier == "perceptron"):
    classifier = perceptron.PerceptronClassifier(legalLabels,options.iterations)
  elif(options.classifier == "neural"):
    if (options.data == "faces"):
      #inputNum, hiddenNum, outputNum, dataNum, lamda
      classifier = neuralNetwork.NeuralNetworkClassifier(legalLabels,FACE_DATUM_WIDTH*FACE_DATUM_HEIGHT,500,1,options.training,0.03)
    else:
      classifier = neuralNetwork.NeuralNetworkClassifier(legalLabels,DIGIT_DATUM_WIDTH*DIGIT_DATUM_HEIGHT,50,10,options.training,3.5)
  else:
    print("Unknown classifier:", options.classifier)
    print(USAGE_STRING)
    
    sys.exit(2)

  args['classifier'] = classifier
  args['featureFunction'] = featureFunction
  args['printImage'] = printImage
  
  return args, options

USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data and performs an odd ratio analysis
                  with label1=3 vs. label2=6
                 """

# Main harness code

def runClassifier(args, options):

  featureFunction = args['featureFunction']
  classifier = args['classifier']
  printImage = args['printImage']
        
  # Load data  
  numTraining = options.training
  numTest = options.test

  if(options.data=="faces"):
    rawTrainingData = samples.loadDataFile("facedata/facedatatrain", numTraining,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTraining)
    combined = list(zip(rawTrainingData, trainingLabels))
    random.shuffle(combined)
    rawTrainingData, trainingLabels = zip(*combined)
    rawValidationData = samples.loadDataFile("facedata/facedatatrain", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTest)
    rawTestData = samples.loadDataFile("facedata/facedatatest", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", numTest)
  else:
    rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
    combined = list(zip(rawTrainingData, trainingLabels))
    random.shuffle(combined)
    rawTrainingData, trainingLabels = zip(*combined)
    rawValidationData = samples.loadDataFile("digitdata/validationimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("digitdata/validationlabels", numTest)
    rawTestData = samples.loadDataFile("digitdata/testimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)
  
  # Extract features
  print("Extracting features...")
  trainingData = list(map(featureFunction, rawTrainingData))  
  validationData = list(map(featureFunction, rawValidationData))
  testData = list(map(featureFunction, rawTestData))
  
  # Conduct training and testing and timing
  start_time = time.time()
  print("Training...")
  accuracies = classifier.train(trainingData, trainingLabels, validationData, validationLabels)
  std_dev = np.std(accuracies)
  print("Standard Deviation of Accuracies over Iterations: ", std_dev)
  print("Validating...")
  guesses = classifier.classify(validationData)
  correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
  correct_percentage = 100.0 * correct / len(validationLabels)
  print("{} correct out of {} ({:.1f}%).".format(correct, len(validationLabels), correct_percentage))
  print("Testing...")
  guesses = classifier.classify(testData)
  end_time = time.time()
  elapsed_time = end_time - start_time
  print("Testing Elapsed Time: {:.2f} seconds".format(elapsed_time))
  correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
  correct_percentage = 100.0 * correct / len(testLabels)
  print("{} correct out of {} ({:.1f}%).".format(correct, len(testLabels), correct_percentage))
  analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)
  

if __name__ == '__main__':
  # Read input
  args, options = readCommand( sys.argv[1:] ) 
  # Run classifier
  runClassifier(args, options)
  
