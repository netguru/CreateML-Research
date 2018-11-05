/*:
 Text classification.
 */
import CreateML
import Foundation

// Initializing the training data from Resources folder.
let trainingDataPath = Bundle.main.path(forResource: "data", ofType: "json", inDirectory: "Data/text/train")!
let trainingData = try! MLDataTable(contentsOf:  URL(fileURLWithPath: trainingDataPath))

// Initializing the classifier with a training data.
let classifier = try! MLTextClassifier(trainingData: trainingData, textColumn: "text", labelColumn: "type")

// Evaluating training & validation accuracies.
let trainingAccuracy = (1.0 - classifier.trainingMetrics.classificationError) * 100
let validationAccuracy = (1.0 - classifier.validationMetrics.classificationError) * 100

// Initializing the properly labeled testing data from Resources folder.
let testingDataPath = Bundle.main.path(forResource: "data", ofType: "json", inDirectory: "Data/text/test")!
let testingData = try! MLDataTable(contentsOf: URL(fileURLWithPath:testingDataPath))

// Counting the testing evaluation.
let evaluationMetrics = classifier.evaluation(on: testingData)
let evaluationAccuracy = (1.0 - evaluationMetrics.classificationError) * 100

// Confusion matrix in order to see which labels were classified wrongly.
let confusionMatrix = evaluationMetrics.confusion
print(confusionMatrix)

// Metadata for saving the model.
let metadata = MLModelMetadata(author: "Author",
                               shortDescription: "A model trained to classify healthy and fast food lunch",
                               version: "1.0")

// Saving the model. Remember to update the path.
//try classifier.write(to: URL(fileURLWithPath: "Path where you would like to save the model"),
//                     metadata: metadata)

/*:
 [Table of contents](Table%20of%20contents) • [Previous page](@previous) • [Next page](@next)
 */
