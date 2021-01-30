const tf = require("@tensorflow/tfjs-node");
const { HyperParams } = require("./HyperParams");
const { createTestData, createTrainData } = require("./generateData");

const trainDataCsvUrl = "file:///home/romain/workspace/tutorial-tensorflowjs-search-hyperparameters/data/train.csv";
const testDataCsvUrl = "file:///home/romain/workspace/tutorial-tensorflowjs-search-hyperparameters/data/test.csv";

async function run() {
  const { dataSet: trainDataset, numOfFeatures } = await createTrainData(trainDataCsvUrl);
  // const { dataSet: validateDataset } = await createDatasetFromCSV(trainDataCsvUrl, {
  //   validate: true,
  // });
  // const { dataSet: testDataset } = await createDatasetFromCSV(testDataCsvUrl, { test: true });

  // Search best hyperparams
  const hyperParams = new HyperParams(
    { dataset: trainDataset, inputSize: numOfFeatures },
    {
      epochs: [100, 300, 500],
      units: [10, 30, 50, 80],
      activation: ["relu"],
      loss: ["meanAbsoluteError", "meanSquaredError"],
      learningRate: [0.01, 0.001, 0.0001],
      // optimizer: [optimizers.adam, optimizers.sgd],
    }
  );
  await hyperParams.findHyperParams();
  const model = hyperParams.getModel();
  console.log(hyperParams.getHyperparams());
/*
{
  bestloss: 0,
  learningRate: 0.001,
  optimizer: 'adam',
  loss: 'meanSquaredError',
  units: 10,
  epochs: 500,
  bestMetric: 0.002963484963402152,
  activation: 'relu'
}

 */
  // const model = tf.sequential();
  // model.add(tf.layers.dense({ inputShape: [numOfFeatures], units: 100, activation: "relu" }));
  // model.add(tf.layers.dense({ units: 100, activation: "relu" }));
  // model.add(tf.layers.dense({ units: 50, activation: "relu" }));
  // model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
  // model.compile({
  //   optimizer: "adam",
  //   loss: "binaryCrossentropy",
  //   // learningRate: 0.001,
  //   metrics: ["accuracy"],
  // });
  // await model.fitDataset(trainDataset, {
  //   epochs: 600,
  //   validationData: validateDataset,
  //   callbacks: {
  //     onEpochEnd: async (epoch, logs) => {
  //       console.log(`Epoch:${epoch} Loss:${logs.loss} Acc:${logs.acc}`);
  //     },
  //   },
  // });
  //
}

run();
