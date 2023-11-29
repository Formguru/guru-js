import {centerCrop, normalize} from "./preprocess.mjs";
import { GURU_KEYPOINTS } from "./core_types.mjs";
export { GURU_KEYPOINTS } from "./core_types.mjs" // for backwards-compatibility

const MODEL_CLASSES = [
  "person",
  "barbell_plates",
];

function arrayMean(array) {
  return arraySum(array) / array.length;
}

export function arrayPeaks(numbers) {
  let maxima = [];
  for (let i = 1; i < numbers.length-1; ++i) {
    if (numbers[i] > numbers[i-1] && numbers[i] > numbers[i+1]) {
      maxima.push(i);
    }
  }
  return maxima;
}

function arraySum(array) {
  return array.reduce((acc, val) => acc + val);
}

function arrayStdDev(arr) {
  const arr_mean = arrayMean(arr);
  const r = function(acc, val) {
    return acc + ((val - arr_mean) * (val - arr_mean))
  };
  return Math.sqrt(arr.reduce(r, 0.0) / arr.length);
}

export function arrayValuesLessThan(arr, cutoff) {
  const pred = [];
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] <= cutoff) {
      pred.push(i);
    }
  }
  return pred;
}

export function arrayVelocities(numbers) {
  const velocity = [0];

  for (let i = 1; i < numbers.length; i++) {
    velocity.push(numbers[i] - numbers[i - 1]);
  }

  return velocity;
}

export function averageKeypointLocation(personFrames, keypoint) {
  let sumX = 0;
  let sumY = 0;
  let sumConfidence = 0;
  let n = 0;

  personFrames.forEach((personFrame) => {
    const keypointPosition = personFrame.keypointLocation(keypoint);

    if (keypointPosition && !isNaN(keypointPosition.x) && !isNaN(keypointPosition.y)) {
      ++n;
      sumX += keypointPosition.x;
      sumY += keypointPosition.y;
      sumConfidence += keypointPosition.confidence;
    }
  });

  return {x: sumX / n, y: sumY / n, confidence: sumConfidence / n};
}

export function averageKeypointLocations(frameObjects, frameStart = 0, frameEnd = 100000000) {
  frameEnd = Math.min(frameObjects.length, frameEnd);
  return new Array(GURU_KEYPOINTS.length).fill(0).map((_, k) => {
    const middleFrameObjects = frameObjects.slice(frameStart, frameEnd);
    const sumX = middleFrameObjects
      .reduce((acc, frameObject) => {
        const keypointLocation = frameObject.keypointLocation(GURU_KEYPOINTS[k]);
        if (keypointLocation) {
          acc += keypointLocation.x;
        }
        return acc;
      }, 0);
    const sumY = middleFrameObjects
      .reduce((acc, frameObject) => {
        const keypointLocation = frameObject.keypointLocation(GURU_KEYPOINTS[k]);
        if (keypointLocation) {
          acc += keypointLocation.y;
        }
        return acc;
      }, 0);
    return [sumX / (frameEnd - frameStart), sumY / (frameEnd - frameStart)];
  });
}

function centerToCornersFormat([centerX, centerY, width, height]) {
  return [
    centerX - width / 2,
    centerY - height / 2,
    centerX + width / 2,
    centerY + height / 2
  ];
}

export function descaleCoords(x, y, originalWidth, originalHeight, scaledWidth, scaledHeight) {
  return [
    x * (originalWidth / scaledWidth) / originalWidth,
    y * (originalHeight / scaledHeight) / originalHeight,
  ];
}

export function ensureArray(maybeArray) {
  return Array.isArray(maybeArray) ? maybeArray : [maybeArray];
}

export function gaussianSmooth(data, sigma) {
  const kernelSize = Math.ceil(sigma * 3) * 2 + 1;
  const kernel = new Array(kernelSize);
  const halfSize = (kernelSize - 1) / 2;

  // Generate the Gaussian kernel
  let sum = 0;
  for (let i = 0; i < kernelSize; i++) {
    const x = i - halfSize;
    kernel[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
    sum += kernel[i];
  }

  // Normalize the kernel
  for (let i = 0; i < kernelSize; i++) {
    kernel[i] /= sum;
  }

  const smoothedData = [];

  // Apply the convolution to the data
  for (let i = 0; i < data.length; i++) {
    let smoothedValue = 0;
    for (let j = 0; j < kernelSize; j++) {
      const dataIndex = i + j - halfSize;
      if (dataIndex >= 0 && dataIndex < data.length) {
        smoothedValue += data[dataIndex] * kernel[j];
      }
    }
    smoothedData.push(smoothedValue);
  }

  return smoothedData;
}

function toRGBFloatArray(imageData) {
  const increment = 4;
  const filteredArray = new Float32Array(imageData.data.length / increment * 3);
  let j = 0;
  for (let i = 0; i < imageData.data.length; i += increment) {
    filteredArray[j++] = imageData.data[i];
    filteredArray[j++] = imageData.data[i + 1];
    filteredArray[j++] = imageData.data[i + 2];
  }
  return filteredArray;
}

/**
 * Determines the most likely class/prediction for each box.
 *
 * @param matrix A [X, Y, 1] matrix, where X is the number of possible prediction classes,
 *  and Y is the number of boxes.
 * @param bboxes Array of objects of size Y, corresponding to the bounding box for each prediction.
 *  @param classes Array of strings that are the classes input to the model for detection.
 * @param threshold The likelihood over which predictions will be counted.
 * @returns An array of length Y, that contains objects that each have a 'probability' field and a
 *  'labelIndex' field. The former is the probability of the highest likelihood prediction for that box,
 *  and labelIndex is the index of the possible classes which that probability is for.
 */
function mostLikelyClass(matrix, bboxes, classes, threshold) {
  const mostLikelyClasses = [];

  const matrixDimensions = matrix.size();
  for (let i = 0; i < matrixDimensions[1]; ++i) {
    let highestProbability = -10000000;
    let highestProbabilityIndex = highestProbability;
    for (let j = 0; j < matrixDimensions[0]; ++j) {
      const nextValue = matrix[j][i][0];
      if (nextValue > highestProbability) {
        highestProbability = nextValue;
        highestProbabilityIndex = j;
      }
    }

    highestProbability = sigmoid(highestProbability);
    if (highestProbability > threshold) {
      mostLikelyClasses.push({
        probability: highestProbability,
        class: classes[highestProbabilityIndex],
        bbox: bboxes[i],
      });
    }
  }

  return mostLikelyClasses;
}

export function movingAverage(arr, windowSize) {
  const slidingDiff = new Array(arr.length).fill(0);
  for (let i = 0; i < arr.length; i++) {
    let sum = 0;
    for (let j = 0; j < windowSize; j++) {
      if (i - j >= 0) {
        sum += arr[i - j];
      }
    }
    slidingDiff[i] = sum / windowSize;
  }
  return slidingDiff;
}

export function normalizeNumbers(timeSeries) {
  if (timeSeries.length < 2) {
    return timeSeries;
  }

  const minValue = Math.min(...timeSeries);
  const maxValue = Math.max(...timeSeries);

  return timeSeries.map((value) => (value - minValue) / (maxValue - minValue));
}

function normalizeImageData(imageData, mean = [0, 0, 0], std = [1, 1, 1]) {
  const data = imageData.data;
  const len = data.length;

  const bytesPerPixel = 4;
  const normalizedArray = new Float32Array(len / bytesPerPixel * 3);
  let j = 0;
  for (let i = 0; i < len; i += bytesPerPixel) {
    let r = data[i];
    let g = data[i + 1];
    let b = data[i + 2];

    if (r > 1.0) {
      r /= 255.0;
    }
    if (g > 1.0) {
      g /= 255.0;
    }
    if (b > 1.0) {
      b /= 255.0;
    }

    normalizedArray[j++] = (r - mean[0]) / std[0];
    normalizedArray[j++] = (g - mean[1]) / std[1];
    normalizedArray[j++] = (b - mean[2]) / std[2];
  }

  return normalizedArray;
}

function scaleImageData(imageData, targetWidth, targetHeight) {
  return scaleImage(
    imageData.data,
    imageData.width,
    imageData.height,
    targetWidth,
    targetHeight
  );
}

function scaleImage(image, sourceWidth, sourceHeight, targetWidth, targetHeight, scalingFactor = 1.0) {
  const scaleX = sourceWidth / targetWidth;
  const scaleY = sourceHeight / targetHeight;

  const bytesPerPixel = 4;
  const targetData = new Float32Array(targetWidth * targetHeight * bytesPerPixel);

  for (let y = 0; y < targetHeight; y++) {
    for (let x = 0; x < targetWidth; x++) {
      const sourceX = Math.floor(x * scaleX);
      const sourceY = Math.floor(y * scaleY);

      const sourceIndex = (sourceY * sourceWidth + sourceX) * bytesPerPixel;
      const targetIndex = (y * targetWidth + x) * bytesPerPixel;

      for (let i = 0; i < bytesPerPixel; i++) {
        targetData[targetIndex + i] = image[sourceIndex + i] * scalingFactor;
      }
    }
  }

  return {
    data: targetData,
    width: targetWidth,
    height: targetHeight,
  };
}

export function selectDetectionClassFromResults(results, detClass) {
  const classLabel = BigInt(MODEL_CLASSES.indexOf(detClass));
  const bestClassIdx = results.labels.data.findIndex((label) => label === classLabel);
  if (bestClassIdx < 0) {
    return [];
  }

  const outputMatrix = tensorToMatrix(results.dets);
  return outputMatrix[0][bestClassIdx];
}

export function lowerCamelToSnakeCase(lowerCamel) {
  return lowerCamel.replace(/([a-z])([A-Z])/g, '$1_$2').toLowerCase();
}

export function snakeToLowerCamelCase(snake) {
  return snake.replace(/_([a-z])/g, function (match, letter) {
    return letter.toUpperCase();
  });
}

/**
 * Implementation of the smoothed z-score algorithm, copied from https://stackoverflow.com/a/57889588.
 *
 * @param {[number]} values - The array of values.
 * @param params - A dictionary containing values for lag, threshold, and influence. May be null.
 * @returns {[number]} - An array, the same length as the input, that contains either 0, 1, or -1, corresponding to the
 *  peaks and troughs.
 */
export function smoothedZScore(values, params) {
  if (values === undefined || values.length === 0) {
    return [];
  }

  const p = params || {};
  const lag = p.lag || 5;
  const threshold = p.threshold || 3.5;
  const influence = p.influence || 0.5;

  if (values.length < lag + 2) {
    throw `y data array too short (${values.length}) for given lag of ${lag}`
  }

  const signals = Array(values.length).fill(0);
  const filteredY = values.slice(0);
  const lead_in = values.slice(0, lag);

  const avgFilter = [];
  avgFilter[lag - 1] = arrayMean(lead_in);
  const stdFilter = [];
  stdFilter[lag - 1] = arrayStdDev(lead_in);

  for (let i = lag; i < values.length; i++) {
    if (Math.abs(values[i] - avgFilter[i - 1]) > (threshold * stdFilter[i - 1])) {
      if (values[i] > avgFilter[i - 1]) {
        signals[i] = +1; // positive signal
      } else {
        signals[i] = -1; // negative signal
      }
      // make influence lower
      filteredY[i] = influence * values[i] + (1 - influence) * filteredY[i - 1];
    } else {
      signals[i] = 0; // no signal
      filteredY[i] = values[i];
    }

    // adjust the filters
    const y_lag = filteredY.slice(i - lag + 1, i + 1);
    avgFilter[i] = arrayMean(y_lag);
    stdFilter[i] = arrayStdDev(y_lag);
  }

  return signals
}

function transposeImageData(imageArray, width, height) {
  const channels = 3;
  const transposed = new Float32Array(channels * height * width);

  let transposedIndex = 0;
  for (let i = 0; i < imageArray.length; i += 3) {
    transposed[transposedIndex++] = imageArray[i];
  }
  for (let i = 1; i < imageArray.length; i += 3) {
    transposed[transposedIndex++] = imageArray[i];
  }
  for (let i = 2; i < imageArray.length; i += 3) {
    transposed[transposedIndex++] = imageArray[i];
  }

  return transposed;
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

export function keypointsFromHeatmap(heatmap, preprocessedFrame, boundingBox) {
  const K = heatmap.length;
  const heatmapHeight = heatmap[0].length;
  const heatmapWidth = heatmap[0][0].length;
  if (![17, 21].includes(K)) {
    throw new Error("Invalid number of keypoints. Expected 17 or 21.");
  }

  function argmax(matrix) {
    if (matrix.length === 0) {
      throw new Error("Cannot perform argmax on an empty matrix.");
    }

    let maxIndex = -1;
    let maxValue = -100000000;

    for (let row = 0; row < heatmapHeight; row++) {
      for (let col = 0; col < heatmapWidth; col++) {
        if (matrix[row][col] > maxValue) {
          maxIndex = (row * heatmapWidth) + col;
          maxValue = matrix[row][col];
        }
      }
    }

    return maxIndex;
  }

  const results = [];
  for (let k = 0; k < K; k++) {
    const max_idx = argmax(heatmap[k]);
    let y = Math.floor(max_idx / heatmapWidth);
    let x = max_idx % heatmapWidth;
    const score = heatmap[k][y][x];

    // descale the coordinates from the dimensions wanted by the model to the dimensions of the bounding box.
    const descaled = descaleCoords(
      (x / heatmapWidth) * preprocessedFrame.newWidth,
      (y / heatmapHeight) * preprocessedFrame.newHeight,
      preprocessedFrame.boundingBoxWidth, preprocessedFrame.boundingBoxHeight,
      preprocessedFrame.newWidth, preprocessedFrame.newHeight,
    );

    // then translate from bounding box coords back to the original image
    results.push([
      boundingBox.topLeft.x + descaled[0] * (preprocessedFrame.boundingBoxWidth / preprocessedFrame.originalWidth),
      boundingBox.topLeft.y + descaled[1] * (preprocessedFrame.boundingBoxHeight / preprocessedFrame.originalHeight),
      score
    ]);
  }

  return results;
}

export function postProcessObjectDetectionResults(results, labels, threshold = 0.01) {
  const numBoxes = results.logits.dims[1];
  const bboxMatrix = tensorToMatrix(results.pred_boxes);
  const bboxes = [];
  for (let i = 0; i < numBoxes; ++i) {
    bboxes.push(centerToCornersFormat([
      bboxMatrix[0][i][0],
      bboxMatrix[0][i][1],
      bboxMatrix[0][i][2],
      bboxMatrix[0][i][3],
    ]));
  }

  return mostLikelyClass(tensorToMatrix(results.logits), bboxes, labels, threshold);
}

export function postProcessPoseEstimationResult(result, keypointTransformer) {
  const keypoints = result.keypoints.data;
  const scores = result.scores.data;
  const transformedKeypoints = [];
  for (let i = 0; i < scores.length; i++) {
    const kpt = { x: keypoints[i * 2], y: keypoints[i * 2 + 1], score: scores[i] };
    const { x, y } = keypointTransformer(kpt);
    kpt.x = x;
    kpt.y = y;
    transformedKeypoints.push(kpt);
  }

  const jointToLocation = {};
  GURU_KEYPOINTS.forEach((keypointName, index) => {
    if (index >= 17) {
      return;
    }
    const { x, y, score } = transformedKeypoints[index];
    jointToLocation[keypointName] = { x, y, confidence: score };
  });
  return jointToLocation;
}

export const preprocessImageForObjectDetection = (imageData, modelWidth, modelHeight) => {
  const resized = scaleImage(
    imageData.data,
    imageData.width, imageData.height,
    modelWidth, modelHeight,
    0.00392156862745098,
  );

  const normalized = normalizeImageData(
    resized,
    [0.48145466, 0.4578275, 0.40821073],
    [0.26862954, 0.26130258, 0.27577711],
  );

  const transposedImage = transposeImageData(normalized, modelWidth, modelHeight);

  return {
    image: transposedImage,
    newWidth: modelWidth,
    newHeight: modelHeight,
  };
};

export const preprocessImageForPersonDetection = (imageData, modelWidth, modelHeight) => {
  const resized = scaleImageData(imageData, modelWidth, modelHeight);
  const rgbImage = toRGBFloatArray(resized);
  const transposedImage = transposeImageData(rgbImage, modelWidth, modelHeight);

  return {
    image: transposedImage,
    newWidth: modelWidth,
    newHeight: modelHeight,
  };
};

export const preprocessImageForPoseEstimation = (imageData, modelWidth, modelHeight, boundingBox) => {
  const processedBBox = {
    x1: boundingBox.topLeft.x * imageData.width,
    y1: boundingBox.topLeft.y * imageData.height,
    x2: boundingBox.bottomRight.x * imageData.width,
    y2: boundingBox.bottomRight.y * imageData.height,
  };

  const cropped = centerCrop(imageData, { inputWidth: modelWidth, inputHeight: modelHeight, boundingBox: processedBBox });
  const normalized = normalize(cropped.image);
  return [new Float32Array(normalized.getData()), cropped];
};

export function preprocessedImageToTensor(ort, preprocessedFrame) {
  return new ort.Tensor(
    'float32',
    preprocessedFrame.image,
    [1, 3, preprocessedFrame.newHeight, preprocessedFrame.newWidth]
  );
}


/**
 * The text batch size for OWL-ViT is 2, so groups the inputs into pairs
 * so that they can be run most efficiently.
 *
 * @param textInputs Array of strings of the items to look for.
 * @returns The Array of Pairs.
 */
export function prepareTextsForOwlVit(textInputs) {
  const paddingValue = "nothing";
  if (Array.isArray(textInputs)) {
    if (textInputs.length % 2 > 0) {
      textInputs.push(paddingValue);
    }

    const pairs = [];
    for (let i = 0; i < textInputs.length; i += 2) {
      pairs.push([textInputs[i], textInputs[i + 1]]);
    }
    return pairs;
  }
  else {
    return [[textInputs, paddingValue]];
  }
}

export function tensorToMatrix(tensor) {
  const dataArray = Array.from(tensor.data);

  function reshapeArray(array, shape) {
    if (shape.length === 0) {
      return array.shift();
    }
    const size = shape.shift();
    const subArray = [];
    for (let i = 0; i < size; i++) {
      subArray.push(reshapeArray(array, shape.slice()));
    }
    return subArray;
  }

  const dimensions = [...tensor.dims];
  const matrix = reshapeArray(dataArray, tensor.dims);
  matrix.size = function() {
    return dimensions;
  };
  return matrix;
}