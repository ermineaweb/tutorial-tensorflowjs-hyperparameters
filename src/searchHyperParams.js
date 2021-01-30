const tf = require("@tensorflow/tfjs-node");
const { trainData, testData } = require("./data");
const { HyperParams, optimizers } = require("./HyperParams");

// Generate data
const train = {
  sizeM2: tf.tensor2d(trainData.sizeM2, [20, 1]),
  price: tf.tensor2d(trainData.price, [20, 1]),
};
const test = {
  sizeM2: tf.tensor2d(testData.sizeM2, [3, 1]),
  price: tf.tensor2d(testData.price, [3, 1]),
};
// Search best hyperparams
const hyperParams = new HyperParams(
  { input: train.sizeM2, label: train.price, inputSize: 1 },
  {
    epochs: [100, 300, 600],
    units: [5, 15, 30, 50, 80],
    loss: ["meanSquaredError"],
    activation: ["relu"],
    learningRate: [0.001, 0.0001, 0.00001],
  }
);

(async () => {
  await hyperParams.findHyperParams();
  const model = hyperParams.getModel();
  console.log(hyperParams.getHyperparams());

  // Evaluation
  model.evaluate(test.sizeM2, test.price).print();

  // Prediction
  model.predict(tf.tensor2d([[100], [115], [130]])).print();
})();
