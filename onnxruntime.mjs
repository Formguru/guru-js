const guruModels = {
  pose: "https://formguru-datasets.s3.us-west-2.amazonaws.com/on-device/20230905/guru-rtmpose-img-256x192.onnx",
  person_detection: "https://formguru-datasets.s3.us-west-2.amazonaws.com/on-device/20230511/tiny-yolov3.onnx",
  object_detection: "https://formguru-datasets.s3.us-west-2.amazonaws.com/on-device/20230718/owl-vit.onnx",
};
var customModels = {};

const getModelInfo = () => {
  return { ...guruModels, ...customModels };
}

const loadModelByUrl = async (modelUrl, sessionOptions) => {
  const cacheName = 'modelCache';
  const cache = await caches.open(cacheName);
  let modelResponse = await cache.match(modelUrl);

  if (modelResponse) {
    console.info(`Using model from cache for ${modelUrl}`);
  } else {
    console.info(`Reading model from URL: ${modelUrl}`);
    modelResponse = await fetch(modelUrl);
    cache.put(modelUrl, modelResponse.clone());
  }

  const modelBytes = await modelResponse.arrayBuffer();
  return await getOrt().InferenceSession.create(modelBytes, sessionOptions);
};

const loadModelByName = async (name, sessionOptions) => {
    const modelInfo = getModelInfo();
    if (!modelInfo[name]) {
        throw new Error(`Model ${name} not found - model names: ${Object.keys(modelInfo)}`);
    }
    return await loadModelByUrl(modelInfo[name], sessionOptions);
}

/**
 * 
 * This style of import works *only* from user-code, where we've already
 * initialized the import map:
 * 
 * import { ort } guru/onnxruntime;
 * 
 * 
 * If you are importing onnxruntime from another ES6 module (i.e., another ES6
 * module definition), you need to use:
 * 
 * import { getOrt } from "./onnxruntime.mjs";
 * ...
 * const ort = getOrt();
 * 
 */

var ort;
var isWebGpuAvailable = false;
const initOrt = (_ort, _customModels = {}, _isWebGpuAvailable = false) => {
  ort = _ort;
  globalThis._ort = _ort;
  customModels = {..._customModels};
  isWebGpuAvailable = _isWebGpuAvailable;
};

const getOrt = () => {
    var ort = globalThis._ort;
    if (!ort) throw new Error("ort not initialized");
    return ort;
};

const listExecutionProviders = () => {
  if (isWebGpuAvailable) {
    return ["webgpu", "wasm"];
  }
  return ["wasm"];
}

export { ort, getOrt, initOrt, loadModelByName, loadModelByUrl, listExecutionProviders }