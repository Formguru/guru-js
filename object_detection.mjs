import { getOrt } from "./onnxruntime.mjs";
import { Box, Position } from "./core_types.mjs";
import { centerCrop } from "./preprocess.mjs";
import { tensorToMatrix } from "./inference_utils.mjs";

export class Detection {
  constructor(label, box, confidence) {
    this.label = label;
    this.box = box;
    this.confidence = confidence;
  }
}

export class YOLOXDetector {
  /**
   * Load a YOLOX detector using the given ONNX model(s).
   *
   * If a single model is given, it is assumed to be the full model.
   * If two models are given, they are assumed to be the backbone and head.
   *
   * @param {InferenceSession|Array[InferenceSession]} onnxModelOrModels
   * @param {Array[string]} categoryNames
   * @param {Object} options
   */
  constructor(
    onnxModelOrModels,
    categoryNames,
    { inputWidth = 640, inputHeight = 640 } = {}
  ) {
    // Assert that the input is either:
    // - a single model
    // - an array of models with length 2
    if (Array.isArray(onnxModelOrModels)) {
      if (onnxModelOrModels.length !== 2) {
        throw new Error("Expected <= 2 models");
      }
      this.models = onnxModelOrModels;
    } else {
      this.models = [onnxModelOrModels];
    }
    this.inputWidth = inputWidth;
    this.inputHeight = inputHeight;
    this.categoryNames = categoryNames;
    this.ort = getOrt();
  }

  async _preprocess(img) {
    const dummyBbox = { x1: 0, y1: 0, x2: img.width - 1, y2: img.height - 1 };
    const resized = centerCrop(img, {
      inputWidth: this.inputWidth,
      inputHeight: this.inputHeight,
      boundingBox: dummyBbox,
      padding: 1.0,
    });
    const nchw = new Float32Array(
      resized.image.getData({ channelsFirst: true })
    );
    const input = new this.ort.Tensor("float32", nchw, [
      1,
      3,
      this.inputHeight,
      this.inputWidth,
    ]);
    return {
      input,
      resized,
    };
  }

  async _forward(input) {
    var _input = { input };
    var output = null;
    for (const model of this.models) {
      output = await model.run(_input);
      _input = output;
    }
    return output;
  }

  async run(img) {
    // Preprocess the input
    const { resized, input } = await this._preprocess(img);

    // Run the model (which may be partitioned)
    const results = await this._forward(input);

    // Build the Detection objects from the output tensor
    const detections = [];
    const [B, N, _] = results.dets.dims;
    const _dets = tensorToMatrix(results.dets);
    const _labels = tensorToMatrix(results.labels);
    for (var b = 0; b < B; b++) {
      for (var n = 0; n < N; n++) {
        const [x1, y1, x2, y2, score] = _dets[b][n];
        const topLeft = resized.reverseTransform({ x: x1, y: y1 });
        const bottomRight = resized.reverseTransform({ x: x2, y: y2 });
        const box = new Box(
          new Position(topLeft.x, topLeft.y, score),
          new Position(bottomRight.x, bottomRight.y, score)
        );
        const labelIndex = _labels[b][n];
        const detection = new Detection(
          this.categoryNames[labelIndex],
          box,
          score
        );
        detections.push(detection);
      }
    }
    return detections;
  }
}