import { getOrt, loadModelByName } from "./onnxruntime.mjs";

class OnnxModel {

  constructor(inputName, session, metadata) {
    this.inputName = inputName;
    this.session = session;
    this.metadata = metadata;
  }

  static async load(modelName, sessionOptions = { executionProviders: ['wasm'] }, metadata) {
    const session = await loadModelByName(modelName, sessionOptions);
    return OnnxModel.loadFromSession(session, metadata);
  }

  static loadFromSession(session, metadata) {
    const inputName = session.inputNames[0];
    const {inputWidth, inputHeight} = {inputWidth: 192, inputHeight: 256, ...metadata};

    const _metadata = new OnnxModelMetadata(
      new ImgNormCfg([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
      new ImgSize(inputWidth, inputHeight)
    );

    return new OnnxModel(inputName, session, _metadata);
  }

  static async loadFromUrl(url, sessionOptions = { executionProviders: ['wasm'] }, metadata) {
    const session = await getOrt().InferenceSession.create(
      url,
      sessionOptions,
    );
    return OnnxModel.loadFromSession(session, metadata);
  }   
}

class ImgNormCfg {
  constructor(mean, std) {
    this.mean = mean;
    this.std = std;
  }
}

class ImgSize {
  constructor(width, height) {
    this.width = width;
    this.height = height;
  }
}

class OnnxModelMetadata {
  constructor(normCfg, size) {
    this.normCfg = normCfg;
    this.size = size;
  }
}

export {OnnxModel, ImgNormCfg, ImgSize, OnnxModelMetadata}
