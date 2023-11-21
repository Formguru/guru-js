import { tensorToMatrix } from "./inference_utils.mjs";
import { Box, Position } from "./core_types.mjs";
import { getOrt } from "./onnxruntime.mjs";
import { resize, crop, copyMakeBorder, normalize } from "./preprocess.mjs";

export class Detection {
  constructor(label, box, confidence) {
    this.label = label;
    this.box = box;
    this.confidence = confidence;
  }
}

export class MixFormerObjectTracker {
  /**
   * @param {InferenceSession} onnxSession
   * @param {Object} options
   */

  constructor(
    onnxSession,
    {
      templateFactor = 2.0,
      templateSize = 112,
      searchFactor = 4.5,
      searchSize = 224,
      updateInterval = 5,
      maxScoreDecay = 1.0,
      minConfidenceThreshold = 0.6,
      maxConsecutiveFailures = 3,
      debug = false,
    } = {}
  ) {
    this.onnxSession = onnxSession;
    this.templateFactor = templateFactor;
    this.templateSize = templateSize;
    this.searchFactor = searchFactor;
    this.searchSize = searchSize;
    this.updateInterval = updateInterval;
    this.maxScoreDecay = maxScoreDecay;
    this.minConfidenceThreshold = minConfidenceThreshold;
    this.maxConsecutiveFailures = maxConsecutiveFailures;

    this.currentXywh = null;
    this.frame_id = 0;
    this.label = "UNKNOWN";
    this.numConsecutiveFailures = 0;
    this.debug = debug;
  }


  /**
   * Initialize the tracker with the initial frame
   * @param {} image The template image
   * @param {*} bbox The bounding-box of the object to track.
   *                 An object with keys x1, y1, x2, y2. The values should be normalized.
   * @param {*} label A string label to identify the object, e.g., "person-1"
   */
  init = async (image, bbox, label = "UNKNOWN") => {
    // Denormalize bounding-box coordinates
    let { x1: x, y1: y, x2: _x2, y2: _y2 } = bbox;
    x *= image.width;
    y *= image.height;
    _x2 *= image.width;
    _y2 *= image.height;

    const [w, h] = [_x2 - x, _y2 - y];
    const xywh = [x, y, w, h];
    const [zPatchArray, _] = this._sampleTarget(
      image,
      xywh,
      this.templateFactor,
      this.templateSize
    );
    if (this.debug) {
      this.rawTemplate = zPatchArray;
      this.rawOnlineTemplate = zPatchArray;
    }
    const template = normalize(zPatchArray);
    this.template = template;
    this.onlineTemplate = template;

    this.maxPredScore = -1.0;
    this.onlineMaxTemplate = template;

    // save states
    this.numConsecutiveFailures = 0;
    this.currentXywh = xywh;
    this.frameId = 0;
    this.label = label;
  };

  /**
   * Returns true if the tracker is currently tracking an object.
   */
  isTracking = () => {
    return (
      this.currentXywh !== null &&
      this.numConsecutiveFailures < this.maxConsecutiveFailures
    );
  };

  /**
   * Update the tracker with the latest image
   * 
   * @param {*} image The latest image
   * @returns { detection: Detection, isTracking: boolean}
   */
  async update(image) {
    if (!this.isTracking()) {
      return { detection: null, isTracking: false };
    }

    const { height: H, width: W } = image;
    this.frameId += 1;

    let [xPatchArr, resizeFactor] = this._sampleTarget(
      image,
      this.currentXywh,
      this.searchFactor,
      this.searchSize
    );
    if (this.debug) {
      this.rawSearch = xPatchArr;
    }

    let search = normalize(xPatchArr);
    let [predBoxes, predScore] = await this._forward(
      this.template,
      this.onlineTemplate,
      search
    );
    let predBox = predBoxes.map((x) => (x * this.searchSize) / resizeFactor);
    this.currentXywh = this._clipBox(
      this._mapBoxBack(predBox, resizeFactor),
      H,
      W,
      10
    );

    this.maxPredScore *= this.maxScoreDecay;

    // update the template
    if (
      predScore > this.minConfidenceThreshold &&
      predScore > this.maxPredScore
    ) {
      let [zPatchArr, _] = this._sampleTarget(
        image,
        this.currentXywh,
        this.templateFactor,
        this.templateSize
      );
      if (this.debug) {
        this.rawOnlineMaxTemplate = zPatchArr;
      }

      this.onlineMaxTemplate = normalize(zPatchArr);
      this.maxPredScore = predScore;
    }

    if (predScore < this.minConfidenceThreshold) {
      this.numConsecutiveFailures += 1;
      return {
        detection: null,
        isTracking: this.isTracking(),
      };
    }

    this.numConsecutiveFailures = 0;

    if (this.frameId % this.updateInterval === 0) {
      if (this.debug) {
        this.rawOnlineTemplate = this.rawOnlineMaxTemplate;
      }
      this.onlineTemplate = this.onlineMaxTemplate;
      this.maxPredScore = -1;
      this.onlineMaxTemplate = this.template;
    }

    // Normalize coordinates
    let [absX, absY, absW, absH] = this.currentXywh;
    const detection = new Detection(
      this.label,
      new Box(
        new Position(absX / W, absY / H, predScore),
        new Position((absX + absW) / W, (absY + absH) / H, predScore)
      ),
      predScore
    );
    return { detection, isTracking: true };
  }

  _sigmoid = (x) => 1 / (1 + Math.exp(-x));

  _forward = async (template, onlineTemplate, search) => {
    const ort = await getOrt();
    const toTensor = ({ width, height, getData }) => {
      const nchw = new Float32Array(getData({ channelsFirst: true }));
      return new ort.Tensor("float32", nchw, [1, 3, height, width]);
    };
    const output = await this.onnxSession.run({
      template: toTensor(template),
      online_template: toTensor(onlineTemplate),
      search: toTensor(search),
    });
    let { boxes, scores } = output;
    boxes = boxes.reshape([4]);
    const score = this._sigmoid(scores.reshape([1]).data[0]);
    return [tensorToMatrix(boxes), score];
  };

  /**
   * Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area
   *  args:
   *      im - cv image
   *      target_bb - target box [x, y, w, h]
   *      search_area_factor - Ratio of crop size to target size
   *      output_sz - (float) Size to which the extracted crop is resized (always square).
   *
   *  returns:
   *      cv image - extracted crop
   *      float - the factor by which the crop has been resized to make the crop size equal output_size
   *
   * @param {*} image
   * @param {*} bbox
   * @param {*} factor
   * @param {*} outputSize
   */
  _sampleTarget = (image, bbox, searchAreaFactor, outputSize) => {
    const [x, y, w, h] = bbox;
    const cropSize = Math.ceil(Math.sqrt(w * h) * searchAreaFactor);
    if (cropSize < 1) {
      throw new Error("Too small bounding box.");
    }

    const x1 = Math.round(x + 0.5 * w - cropSize * 0.5);
    const x2 = x1 + cropSize;

    const y1 = Math.round(y + 0.5 * h - cropSize * 0.5);
    const y2 = y1 + cropSize;

    const { width, height } = image;

    const leftPad = Math.max(0, -x1);
    const rightPad = Math.max(x2 - width + 1, 0);

    const topPad = Math.max(0, -y1);
    const bottomPad = Math.max(y2 - height + 1, 0);

    var _cropped = crop(image, {
      x1: x1 + leftPad,
      y1: y1 + topPad,
      x2: x2 - rightPad,
      y2: y2 - bottomPad,
    });

    var cropped = copyMakeBorder(_cropped, {
      top: topPad,
      bottom: bottomPad,
      left: leftPad,
      right: rightPad,
    });
    const resizeFactor = outputSize / cropSize;
    const output = resize(cropped, { width: outputSize, height: outputSize });
    return [output, resizeFactor];
  };

  // Map the outpout back to the original image coordinates
  _mapBoxBack(predBox, resizeFactor) {
    let cxPrev = this.currentXywh[0] + 0.5 * this.currentXywh[2];
    let cyPrev = this.currentXywh[1] + 0.5 * this.currentXywh[3];

    let [cx, cy, w, h] = predBox;
    let halfSide = (0.5 * this.searchSize) / resizeFactor;
    let cxReal = cx + (cxPrev - halfSide);
    let cyReal = cy + (cyPrev - halfSide);

    return [cxReal - 0.5 * w, cyReal - 0.5 * h, w, h];
  }

  _clipBox(box, H, W, margin = 0) {
    var [x1, y1, w, h] = box;
    var [x2, y2] = [x1 + w, y1 + h];
    x1 = Math.min(Math.max(0, x1), W - margin);
    x2 = Math.min(Math.max(margin, x2), W);
    y1 = Math.min(Math.max(0, y1), H - margin);
    y2 = Math.min(Math.max(margin, y2), H);
    w = Math.max(margin, x2 - x1);
    h = Math.max(margin, y2 - y1);
    return [x1, y1, w, h];
  }
}
