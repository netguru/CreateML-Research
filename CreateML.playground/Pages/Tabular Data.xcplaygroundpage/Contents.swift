/*:
 Tabular data classification.
 */
import CreateML
import Foundation

// Initializing the data from Resources folder.
let dataPath = Bundle.main.path(forResource: "data", ofType: "json", inDirectory: "Data/table")!
let dataTable = try! MLDataTable(contentsOf: URL(fileURLWithPath: dataPath))

// Spliting the data into training and testing ones.
let (trainingData, testingData) = dataTable.randomSplit(by: 0.8, seed: 5)

// Initializing the classifier with a training data.
let classifier = try! MLClassifier(trainingData: trainingData, targetColumn: "type")

// The alternative way of initializing manually a concrete classifier.
/*
var modelParameters = MLBoostedTreeClassifier.ModelParameters()
modelParameters.maxIterations = 20
let classifier = try! MLBoostedTreeClassifier(trainingData: trainingData, targetColumn: "type", featureColumns: nil, parameters: modelParameters)
*/

// Evaluating training & validation accuracies.
let trainingAccuracy = (1.0 - classifier.trainingMetrics.classificationError) * 100
let validationAccuracy = (1.0 - classifier.validationMetrics.classificationError) * 100

// Initializing the properly labeled testing data from Resources folder.
let evaluationMetrics = classifier.evaluation(on: testingData)
let evaluationAccuracy = (1.0 - evaluationMetrics.classificationError) * 100

// Metadata for saving the model.
let metadata = MLModelMetadata(author: "Author",
                               shortDescription: "A model trained to classify healthy and fast food lunch",
                               version: "1.0")
// Saving the model. Remember to update the path.
try classifier.write(to: URL(fileURLWithPath: "Path where you would like to save the model"),
                     metadata: metadata)
/*:
 [Table of contents](Table%20of%20contents) • [Previous page](@previous)
 */
