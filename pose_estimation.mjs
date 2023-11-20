import { getOrt } from "./onnxruntime.mjs";
import { Position, GURU_KEYPOINTS } from "./core_types.mjs";
import { centerCrop, normalize } from "./preprocess.mjs";
import { tensorToMatrix } from "./inference_utils.mjs";

export class GuruPoseEstimator {
  /**
   * Load the GuruRTM image-based pose model using the given ONNX model(s).
   *
   * If a single model is given, it is assumed to be the full model.
   * If two models are given, they are assumed to be the backbone and head.
   *
   * @param {InferenceSession|Array[InferenceSession]} onnxModelOrModels
   * @param {Object} options
   */
  constructor(
    onnxModelOrModels,
    {
      keypointNames = [...GURU_KEYPOINTS].slice(0, 17),
      inputWidth = 192,
      inputHeight = 256,
    } = {}
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
    this.keypointNames = keypointNames;
    this.ort = getOrt();
  }

  async _preprocess(image, boundingBox) {
    const {
      topLeft: { x: x1, y: y1 },
    } = boundingBox;
    const {
      bottomRight: { x: x2, y: y2 },
    } = boundingBox;
    const maxNormalizedRange = 2.0;
    if (x2 - x1 > maxNormalizedRange || y2 - y1 > maxNormalizedRange) {
      throw new Error("boundingBox is not normalized");
    }
    const bbox = {
      x1: x1 * image.width,
      y1: y1 * image.height,
      x2: x2 * image.width,
      y2: y2 * image.height,
    };

    const opts = {
      inputWidth: this.inputWidth,
      inputHeight: this.inputHeight,
      boundingBox: bbox,
    };
    var cropped = centerCrop(image, opts);
    var normalized = normalize(cropped.image);
    const nchw = new Float32Array(normalized.getData());
    const input = new this.ort.Tensor("float32", nchw, [
      1,
      3,
      this.inputHeight,
      this.inputWidth,
    ]);
    return {
      input,
      cropped,
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

  async run(img, boundingBox) {
    const _arrayEq = (a, b) => {
      return (
        Array.isArray(a) &&
        Array.isArray(b) &&
        a.length === b.length &&
        a.every((val, index) => val === b[index])
      );
    };

    const { input, cropped } = await this._preprocess(img, boundingBox);
    let { keypoints, scores } = await this._forward(input);
    const K = this.keypointNames.length;
    if (!_arrayEq(keypoints.dims, [1, 1, K, 2])) {
      throw new Error(
        `Expected dims [1, 1, ${K}, 2] but got ${keypoints.dims}`
      );
    }
    if (!_arrayEq(scores.dims, [1, 1, K])) {
      throw new Error(`Expected dims [1, 1, ${K}] but got ${scores.dims}`);
    }

    keypoints = tensorToMatrix(keypoints)[0][0];
    scores = tensorToMatrix(scores)[0][0];

    const j2p = {};
    for (let i = 0; i < K; i++) {
      const [_x, _y] = keypoints[i];
      const { x, y } = cropped.reverseTransform({ x: _x, y: _y });
      j2p[GURU_KEYPOINTS[i]] = new Position(x, y, scores[i]);
    }
    return j2p;
  }
}
