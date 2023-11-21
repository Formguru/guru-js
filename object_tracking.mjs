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
   * @param {InferenceSession} onnxModel
   * @param {Object} options
   */

  constructor(
    onnxModel,
    {
      templateFactor = 2.0,
      templateSize = 112,
      searchFactor = 4.5,
      searchSize = 224,
      updateInterval = 10,
      maxScoreDecay = 1.0,
    } = {}
  ) {
    this.onnxModel = onnxModel;
    this.templateFactor = templateFactor;
    this.templateSize = templateSize;
    this.searchFactor = searchFactor;
    this.searchSize = searchSize;
    this.updateInterval = updateInterval;
    this.maxScoreDecay = maxScoreDecay;

    this.state = null;
    this.frame_id = 0;
    this.label = null;
  }

  sigmoid = (x) => 1 / (1 + Math.exp(-x));

  _forward = async (template, onlineTemplate, search) => {
    const ort = await getOrt();
    const toTensor = ({width, height, getData}) => {
        const nchw = new Float32Array(getData({ channelsFirst: true }));
        return new ort.Tensor("float32", nchw, [ 1, 3, height, width ]);
    }
    console.log("Running inference...");
    const output = await this.onnxModel.run({
        template: toTensor(template),
        online_template: toTensor(onlineTemplate),
        search: toTensor(search),
    });
    console.log(`output = ${JSON.stringify(output)}`);
    let { boxes, scores } = output;
    console.log(`boxes = ${JSON.stringify(boxes)}`);
    console.log(`scores = ${JSON.stringify(scores)}`);
    boxes = boxes.reshape([4]);
    const score = this.sigmoid(scores.reshape([1]).data[0]);
    return [tensorToMatrix(boxes), score];
  }

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
   * @param {*} output_sz
   */
  _sampleTarget = (image, bbox, search_area_factor, output_sz) => {
    const [x, y, w, h] = bbox;
    // crop image
    const crop_sz = Math.ceil(Math.sqrt(w * h) * search_area_factor);
    console.log(`w = ${w}, h = ${h}; crop_sz = ${crop_sz}`);
    if (crop_sz < 1) {
      throw new Error("Too small bounding box.");
    }

    const x1 = Math.round(x + 0.5 * w - crop_sz * 0.5);
    const x2 = x1 + crop_sz;

    const y1 = Math.round(y + 0.5 * h - crop_sz * 0.5);
    const y2 = y1 + crop_sz;

    const { width, height } = image;

    const x1_pad = Math.max(0, -x1);
    const x2_pad = Math.max(x2 - width + 1, 0);

    const y1_pad = Math.max(0, -y1);
    const y2_pad = Math.max(y2 - height + 1, 0);

    console.log(
      `centerCrop: {${x1}, ${y1}, ${x2}, ${y2}}; padding: ${x1_pad}, ${y1_pad}, ${x2_pad}, ${y2_pad}; crop_sz = ${crop_sz}`
    );
    var _cropped = crop(image,
      { x1: x1 + x1_pad, y1: y1 + y1_pad, x2: x2 - x2_pad, y2: y2 - y2_pad },
    );
    this._croppedRaw = resize(_cropped, { width: output_sz, height: output_sz });
    var cropped = copyMakeBorder(_cropped, {
      top: y1_pad,
      bottom: y2_pad,
      left: x1_pad,
      right: x2_pad,
    });
    const resizeFactor = output_sz / crop_sz;
    const output = resize(cropped, { width: output_sz, height: output_sz });
    return [output, resizeFactor];
  };

  _preprocess = (image) => {
    return normalize(image);
  };

  init = async (image, bbox, label) => {
    const { x1: x, y1: y, x2: _x2, y2: _y2 } = bbox;
    const [w, h] = [_x2 - x, _y2 - y];
    const xywh = [x, y, w, h];
    const [zPatchArray, _] = this._sampleTarget(
      image,
      xywh,
      this.templateFactor,
      this.templateSize
    );
    this.rawTemplate = zPatchArray;
    const template = this._preprocess(zPatchArray);
    this.template = template;
    this.rawOnlineTemplate = zPatchArray;
    this.onlineTemplate = template;

    this.maxPredScore = -1.0;
    this.onlineMaxTemplate = template;

    // save states
    this.state = xywh;
    this.frameId = 0;
    this.label = label;
  };

  async update(image) {
    const {height: H, width: W} = image;
    this.frameId += 1;

    let [xPatchArr, resizeFactor] = this._sampleTarget(
      image,
      this.state,
      this.searchFactor,
      this.searchSize
    );
    this.rawSearch = xPatchArr;

    let search = this._preprocess(xPatchArr);
    let [predBoxes, predScore] = await this._forward(
      this.template,
      this.onlineTemplate,
      search
    );
    let predBox = predBoxes.map((x) => (x * this.searchSize) / resizeFactor);
    console.log(`[FRAME ${this.frameId}] predBox = ${JSON.stringify(predBox)}`);
    console.log(`[FRAME ${this.frameId}] predScore = ${JSON.stringify(predScore)}`);

    this.state = this._clipBox(this._mapBoxBack(predBox, resizeFactor), H, W, 10);
    console.log(`[FRAME ${this.frameId}] this.state = ${JSON.stringify(this.state)}`);

    this.maxPredScore *= this.maxScoreDecay;

    if (predScore > 0.5 && predScore > this.maxPredScore) {
      let [zPatchArr, _] = this._sampleTarget(
        image,
        this.state,
        this.templateFactor,
        this.templateSize
      );
      this.rawOnlineMaxTemplate = zPatchArr;

      this.onlineMaxTemplate = this._preprocess(zPatchArr);
      this.maxPredScore = predScore;
    }

    if (this.frameId % this.updateInterval === 0) {
      this.rawOnlineTemplate = this.rawOnlineMaxTemplate;
      this.onlineTemplate = this.onlineMaxTemplate;
      this.maxPredScore = -1;
      this.onlineMaxTemplate = this.template;
    }

    let [absX, absY, absW, absH] = this.state;
    return new Detection(
        this.label,
        new Box(
            new Position(absX / W, absY / H, predScore),
            new Position((absX + absW) / W, (absY + absH) / H, predScore),
        ),
        predScore,
    );
  }

  _mapBoxBack(predBox, resizeFactor) {
    let cxPrev = this.state[0] + 0.5 * this.state[2];
    let cyPrev = this.state[1] + 0.5 * this.state[3];

    let [cx, cy, w, h] = predBox;
    let halfSide = (0.5 * this.searchSize) / resizeFactor;
    let cxReal = cx + (cxPrev - halfSide);
    let cyReal = cy + (cyPrev - halfSide);

    return [cxReal - 0.5 * w, cyReal - 0.5 * h, w, h];
  }

  _clipBox(box, H, W, margin=0) {
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
