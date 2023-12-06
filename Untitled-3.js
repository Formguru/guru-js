import { resize } from "guru/preprocess";
import { getOrt, loadModelByName } from "guru/onnxruntime";
import { tensorToMatrix } from "guru/inference_utils";
import { drawRect } from "guru/draw";

const preprocessImage = (image) => {
  return resize(image, { width: 320, height: 240 });
};

const drawResizedImage = (image, context) => {
  const sourcePixels = image.getData({ channelsFirst: false });
  const targetImageData = context.createImageData(image.height, image.width);
  console.log("image: ", image);
  console.log("sourcePixels: ", sourcePixels);
  console.log("targetImageData: ", targetImageData);
  for (var row = 0; row < image.height; row += 1) {
    for (var col = 0; col < image.width; col += 1) {
      const sourceIndex = (row * image.width + col) * image.numChannels;
      const targetIndex = (row * image.width + col) * 4;
      for (var channel = 0; channel < image.numChannels; channel += 1) {
        targetImageData.data[targetIndex + channel] =
          sourcePixels[sourceIndex + channel];
      }
      targetImageData.data[targetIndex + 3] = 255;
    }
  }

  context.putImageData(targetImageData, 0, 0);
};

const drawImage = (image, context) => {
  context.putImageData(image, 0, 0);
};

const normalize = (image) => {
  const { width, height, numChannels } = image;
  const chw = image.getData({ channelsFirst: true });
  const output = new Float32Array(3 * height * width);

  // if input is integer in [0,255], convert to float32 in [0,1]
  const normFactor = 128;

  // rgb
  const mean = [127, 127, 127];
  // const mean = [0.48145466, 0.4578275, 0.40821073];
  // const std = [0.26862954, 0.26130258, 0.27577711];

  for (let i = 0; i < chw.length; i += numChannels) {
    output[i + 0] = (chw[i + 0] - mean[0]) / normFactor;
    output[i + 1] = (chw[i + 1] - mean[1]) / normFactor;
    output[i + 2] = (chw[i + 2] - mean[2]) / normFactor;
  }

  return output;
};

export default class GuruSchema {
  constructor() {
    this.resizedFrames = [];
    this.frames = [];
    this.ort = getOrt();
    this.model = null;
    this.faces = [];
  }

  async getModel() {
    if (this.model === null) {
      this.model = await loadModelByName("version-RFB-320.onnx");
    }
    return this.model;
  }

  async processFrame(frame) {
    const resizedImage = preprocessImage(frame.image);
    const modelInput = new this.ort.Tensor("float32", normalize(resizedImage), [
      1,
      3,
      resizedImage.height,
      resizedImage.width,
    ]);

    const model = await this.getModel();
    const modelOutput = await model.run({ input: modelInput });
    // console.log("modelOutput: ", modelOutput);
    const boxesMatrix = tensorToMatrix(modelOutput.boxes);
    // console.log("boxesMatrix: ", boxesMatrix);

    //boxes = [1, 4420, 4]
    //scores = [1, 4420, 2]
    const scoresMatrix = tensorToMatrix(modelOutput.scores);

    const proposals = scoresMatrix[0];

    console.assert(proposals.length === 4420);

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
    console.log("highestScoreIndex: ", highestScoreIndex);

    this.faces.push({
      box: boxesMatrix[0][highestScoreIndex],
      timestamp: frame.timestamp,
    });

    return this.outputs();
  }

  renderFrame(frameCanvas) {
    const currentFace = this.faces.find(
      (face) => face.timestamp >= frameCanvas.timestamp
    );
    console.log("currentFace: ", currentFace);

    if (currentFace) {
      drawRect(
        frameCanvas.canvas,
        {
          x: currentFace.box[0] * frameCanvas.canvas.canvas.width,
          y: currentFace.box[1] * frameCanvas.canvas.canvas.height,
        },
        {
          x: currentFace.box[2] * frameCanvas.canvas.canvas.width,
          y: currentFace.box[3] * frameCanvas.canvas.canvas.height,
        },
        { color: "red" }
      );
    }

    // drawResizedImage(this.resizedFrames[0], frameCanvas.canvas);
    // drawImage(this.frames[0], frameCanvas.canvas);
  }

  async outputs() {
    return {};
  }
}
