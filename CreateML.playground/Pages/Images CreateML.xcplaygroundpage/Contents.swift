/*:
 That's an example of a bit more advanced way to create a model, in comparison to CreateMLUI method.
 */

import CreateML
import Foundation

// Initializing the properly labeled training data from Resources folder.
let trainingDataPath = Bundle.main.path(forResource: "train", ofType: nil, inDirectory: "Data/image")!
let trainingData = MLImageClassifier.DataSource.labeledDirectories(at: URL(fileURLWithPath: trainingDataPath))

// Initializing the classifier with a training data.
let classifier = try! MLImageClassifier(trainingData: trainingData)

// Evaluating training & validation accuracies.
let trainingAccuracy = (1.0 - classifier.trainingMetrics.classificationError) * 100
let validationAccuracy = (1.0 - classifier.validationMetrics.classificationError) * 100

// Initializing the properly labeled testing data from Resources folder.
let testingDataPath = Bundle.main.path(forResource: "test", ofType: nil, inDirectory: "Data/image")!
let testingData = MLImageClassifier.DataSource.labeledDirectories(at: URL(fileURLWithPath: testingDataPath))

// Counting the testing evaluation.
let evaluationMetrics = classifier.evaluation(on: testingData)
let evaluationAccuracy = (1.0 - evaluationMetrics.classificationError) * 100

// Confusion matrix in order to see which labels were classified wrongly.
let confusionMatrix = evaluationMetrics.confusion
print("Confusion matrix: \(confusionMatrix)")

// Metadata for saving the model.
let metadata = MLModelMetadata(author: "Author",
                               shortDescription: "A model trained to classify healthy and fast food lunch",
                               version: "1.0")

// Saving the model. Remember to update the path. 
try classifier.write(to: URL(fileURLWithPath: "Path where you would like to save the model"),
                     metadata: metadata)

/*:
 [Table of contents](Table%20of%20contents) • [Previous page](@previous) • [Next page](@next)
 */
