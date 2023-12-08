import { resize } from "./preprocess.mjs";
import { getOrt } from "./onnxruntime.mjs";
import { tensorToMatrix } from "./inference_utils.mjs";

export class GuruFaceDetector {
  /**
   * Initializes the GuruFaceDetector for face detection using a pre-trained ONNX model.
   *
   * The constructor expects a 'guruModel' object which contains the ONNX model session
   * specifically trained for face detection tasks. This model is utilized to perform
   * inference on the input images. Additionally, an ONNX Runtime (ORT) instance is
   * retrieved and stored for further use in the model inference process.
   *
   * @param {Object} guruModel - An object containing the ONNX model session.
   *                             This is typically loaded from an external ONNX model file.
   */
  constructor(guruModel) {
    this.model = guruModel.session;
    this.ort = getOrt();
  }

  _preprocessImage(image) {
    return resize(image, { width: 320, height: 240 });
  }

  _normalize(image) {
    const { width, height, numChannels } = image;
    const chw = image.getData({ channelsFirst: true });
    const output = new Float32Array(3 * height * width);

    // if input is integer in [0,255], convert to float32 in [0,1]
    const normFactor = 128;

    const mean = [127, 127, 127];

    for (let i = 0; i < chw.length; i += numChannels) {
      output[i + 0] = (chw[i + 0] - mean[0]) / normFactor;
      output[i + 1] = (chw[i + 1] - mean[1]) / normFactor;
      output[i + 2] = (chw[i + 2] - mean[2]) / normFactor;
    }

    return output;
  }

  async run(img) {
    const resizedImage = this._preprocessImage(img);
    const modelInput = new this.ort.Tensor(
      "float32",
      this._normalize(resizedImage),
      [1, 3, resizedImage.height, resizedImage.width]
    );

    const modelOutput = await this.model.run({ input: modelInput });
    const boxesMatrix = tensorToMatrix(modelOutput.boxes);
    const scoresMatrix = tensorToMatrix(modelOutput.scores);

    const proposals = scoresMatrix[0];

    var highestScore = 0;
    var highestScoreIndex = 0;

    for (let i = 0; i < proposals.length; i++) {
      const proposal = proposals[i];
      const score = proposal[1];
      if (score > highestScore) {
        highestScore = score;
        highestScoreIndex = i;
      }
    }

    console.log("highestScore: ", highestScore);

    return {
      box: boxesMatrix[0][highestScoreIndex],
    };
  }
}
