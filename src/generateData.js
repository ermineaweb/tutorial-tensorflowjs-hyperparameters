const tf = require("@tensorflow/tfjs-node");

async function createTrainData(url) {
  const columnConfigs = {
    Price: { isLabel: true },
    Size: { isLabel: false },
  };
  const numOfFeatures = Object.values(columnConfigs).length - 1;
  const dataset = tf.data.csv(url, {
    columnConfigs,
    configuredColumnsOnly: true,
  });

  const flattenedDataset = dataset
    .shuffle(40)
    .map(({ xs, ys }) => {
      return { xs: Object.values(xs), ys: Object.values(ys) };
    })
    .batch(5)
    .map(({ xs, ys }) => {
      return { xs: xs.sub(xs.min()).div(xs.max().sub(xs.min())), ys: ys.sub(ys.min()).div(ys.max().sub(ys.min())) };
    });

  return { dataSet: flattenedDataset, numOfFeatures };
}

async function createTestData(url) {
  const columnConfigs = {
    Price: { isLabel: true },
    Size: { isLabel: false },
  };
  const numOfFeatures = Object.values(columnConfigs).length - 1;

  const dataset = tf.data.csv(url, {
    columnConfigs,
    configuredColumnsOnly: true,
  });

  const flattenedDataset = dataset
    // .take(3)
    .map((xs) => Object.values(xs))
    .batch(1);

  return { dataSet: flattenedDataset, numOfFeatures };
}

function normalizeMeanStd(value, mean, std) {
  return (value - mean) / std;
}

function normalizeMinMax(value, min, max) {
  return (value - min) / (max - min);
}

async function meanAndStdDevOfDatasetRow(dataset, columnName) {
  let totalSamples = 0;
  let sum = 0;
  let mean = 0;
  let squareDiffFromMean = 0;

  await dataset.forEachAsync(({ xs, ys }) => {
    const x = xs[columnName];
    if (x != null || x !== undefined) {
      totalSamples += 1;
      sum += x;
      mean = sum / totalSamples;
      squareDiffFromMean = (mean - x) * (mean - x);
    }
  });

  const variance = squareDiffFromMean / totalSamples;
  const std = Math.sqrt(variance);

  return { mean, std };
}

async function minMax(dataset, columnName) {
  let min = 0;
  let max = 0;

  await dataset.forEachAsync(({ xs, ys }) => {
    if (xs[columnName] < min) min = xs[columnName];
    if (xs[columnName] > max) max = xs[columnName];
  });

  return { min, max };
}

async function minMaxTest(dataset, columnName) {
  let min = 0;
  let max = 0;

  await dataset.forEachAsync((xs) => {
    if (xs[columnName] < min) min = xs[columnName];
    if (xs[columnName] > max) max = xs[columnName];
  });

  return { min, max };
}

async function meanAndStdDevOfDatasetRowTest(dataset, columnName) {
  let totalSamples = 0;
  let sum = 0;
  let mean = 0;
  let squareDiffFromMean = 0;

  await dataset.forEachAsync((xs) => {
    const x = xs[columnName];
    if (x != null || x !== undefined) {
      totalSamples += 1;
      sum += x;
      mean = sum / totalSamples;
      squareDiffFromMean = (mean - x) * (mean - x);
    }
  });

  const variance = squareDiffFromMean / totalSamples;
  const std = Math.sqrt(variance);

  return { mean, std };
}

function normalizeDataset() {
  let sampleSoFar = 0;
  let sumSoFar = 0;
  return (x) => {
    sampleSoFar += 1;
    sumSoFar += x;
    const estimatedMean = sumSoFar / sampleSoFar;
    return x - estimatedMean;
  };
}

module.exports = { createTestData, createTrainData };
