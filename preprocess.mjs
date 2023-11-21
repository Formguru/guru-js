const Mat = ({ data, width, height, format }) => {
  if (!(height > 0 && width > 0)) {
    throw new Error(`Invalid dimensions: ${width}x${height}`);
  }
  if (data.length === 0) {
    throw new Error("Empty data array");
  }

  const validFormats = ["RGBA", "RGB"];
  const numChannels = format.length;

  if (!validFormats.includes(format)) {
    throw new Error(`Invalid format: ${format}`);
  }

  const hwc = data;
  // data dimensions are height, width, channels
  const delta = Math.abs(hwc.length - width * height * numChannels);
  if (delta > 0.01) {
    throw new Error(
      `Expected array size to be ${width * height * numChannels} but got ${
        hwc.length
      }`
    );
  }

  const convertTo = (targetFormat) => {
    if (format === targetFormat) {
      return Mat({ data, width, height, format });
    }
    if (format !== "RGBA") {
      throw new Error("Only RGBA to RGB conversion is implemented so far");
    }
    const outputChannels = 3;
    var output;
    if (data instanceof Uint8ClampedArray) {
      output = new Uint8ClampedArray(outputChannels * height * width);
    } else if (data instanceof Float32Array) {
      output = new Float32Array(outputChannels * height * width);
    } else {
      throw new Error(`Unsupported data type: ${typeof data}`);
    }

    const targetPixelOffset = (srcPixelOffset) => {
      return Math.floor(srcPixelOffset / numChannels) * outputChannels;
    };

    // copy the first 3 channels from data to output
    for (let i = 0; i < hwc.length; i += numChannels) {
      let j = targetPixelOffset(i);
      output[j] = hwc[i];
      output[j + 1] = hwc[i + 1];
      output[j + 2] = hwc[i + 2];
    }

    return Mat({ data: output, width, height, format: targetFormat });
  };

  const toChannelsFirst = () => {
    const chw = new Float32Array(numChannels * height * width);
    let transposedIndex = 0;
    for (let channel = 0; channel < numChannels; channel++) {
      for (let i = channel; i < hwc.length; i += numChannels) {
        chw[transposedIndex++] = hwc[i];
      }
    }
    return chw;
  };

  const getData = ({ channelsFirst = true } = {}) => {
    return channelsFirst ? toChannelsFirst() : hwc;
  };

  return {
    format,
    width,
    height,
    numChannels,
    getData,
    convertTo,
  };
};

const ensureRgbMat = (imgOrMat) => {
  var mat = null;
  if (imgOrMat.getData) {
    mat = imgOrMat;
  } else {
    const numChannels = Math.round(
      imgOrMat.data.length / (imgOrMat.width * imgOrMat.height)
    );
    if (!(numChannels === 3 || numChannels === 4)) {
      throw new Error(`Unsupported number of channels: ${numChannels}`);
    }

    mat = Mat({
      data: imgOrMat.data,
      height: imgOrMat.height,
      width: imgOrMat.width,
      format: numChannels === 4 ? "RGBA" : "RGB",
    });
  }
  if (mat.format !== "RGB") {
    mat = mat.convertTo("RGB");
  }
  return mat;
};

const scaleBox = (bbox, scale, imgWidth, imgHeight) => {
  let { x1, y1, x2, y2 } = bbox;
  let boxWidth = x2 - x1;
  let boxHeight = y2 - y1;
  let newWidth = boxWidth * scale;
  let newHeight = boxHeight * scale;
  x1 -= (newWidth - boxWidth) / 2.0;
  y1 -= (newHeight - boxHeight) / 2.0;
  x2 = x1 + newWidth;
  y2 = y1 + newHeight;

  // clamp to image bounds
  x1 = Math.max(0, x1);
  y1 = Math.max(0, y1);
  x2 = Math.min(x2, imgWidth);
  y2 = Math.min(y2, imgHeight);
  return { x1, y1, x2, y2 };
};

const crop = (image, { x1, y1, x2, y2 }) => {
  image = ensureRgbMat(image);
  const src = image.getData({ channelsFirst: false });
  const ArrType =
    src instanceof Uint8ClampedArray ? Uint8ClampedArray : Float32Array;
  const [width, height] = [x2 - x1, y2 - y1];
  const data = new ArrType(width * height * image.numChannels);
  for (var row = 0; row < height; row++) {
    for (var col = 0; col < width; col++) {
      for (var channel = 0; channel < image.numChannels; channel++) {
        const srcIdx =
          ((y1 + row) * image.width + (x1 + col)) * image.numChannels + channel;
        const dstIdx = (row * width + col) * image.numChannels + channel;
        data[dstIdx] = src[srcIdx];
      }
    }
  }
  return Mat({
    data,
    width,
    height,
    format: image.format,
  });
};

const centerCrop = (
  image,
  {
    boundingBox,
    inputWidth: targetWidth,
    inputHeight: targetHeight,
    padding = 1.25,
  }
) => {
  image = ensureRgbMat(image);
  const rgbArr = image.getData({ channelsFirst: false });
  const { x1, y1, x2, y2 } = scaleBox(
    boundingBox,
    padding,
    image.width,
    image.height
  );
  const [cropWidth, cropHeight] = [x2 - x1, y2 - y1];
  const outputChannels = image.numChannels;

  // Initialize output array with zeros for padding
  var output;
  if (rgbArr instanceof Uint8ClampedArray) {
    output = new Uint8ClampedArray(outputChannels * targetHeight * targetWidth);
  } else if (rgbArr instanceof Float32Array) {
    output = new Float32Array(outputChannels * targetHeight * targetWidth);
  } else {
    throw new Error(`Unsupported data type: ${typeof rgbArr}`);
  }
  output = output.fill(0);
  const scale = Math.min(targetWidth / cropWidth, targetHeight / cropHeight);

  // Calculate center of cropped area and target area
  const centerCropX = x1 + cropWidth / 2;
  const centerCropY = y1 + cropHeight / 2;
  const centerTargetX = targetWidth / 2;
  const centerTargetY = targetHeight / 2;

  // Iterate over target image dimensions
  for (let y = 0; y < targetHeight; y++) {
    for (let x = 0; x < targetWidth; x++) {
      // Map target coordinates to input coordinates
      const srcX = Math.round(centerCropX + (x - centerTargetX) / scale);
      const srcY = Math.round(centerCropY + (y - centerTargetY) / scale);

      // Check if the mapped coordinates are within the crop bounds
      if (
        srcX >= x1 &&
        srcX < x1 + cropWidth &&
        srcY >= y1 &&
        srcY < y1 + cropHeight
      ) {
        const targetIdx = (y * targetWidth + x) * outputChannels;
        const srcIdx = (srcY * image.width + srcX) * image.numChannels;

        // Copy the pixel
        output[targetIdx] = rgbArr[srcIdx];
        output[targetIdx + 1] = rgbArr[srcIdx + 1];
        output[targetIdx + 2] = rgbArr[srcIdx + 2];
        if (outputChannels === 4) {
          // alpha channel
          output[targetIdx + 3] = rgbArr[srcIdx + 3];
        }
      }
    }
  }

  const xOffset = targetWidth / 2 - centerCropX * scale;
  const yOffset = targetHeight / 2 - centerCropY * scale;

  return {
    image: Mat({
      data: output,
      width: targetWidth,
      height: targetHeight,
      format: "RGB",
    }),
    reverseTransform: (coords) => {
      let { x, y } = coords;
      x -= xOffset;
      x /= scale;
      x /= image.width;

      y -= yOffset;
      y /= scale;
      y /= image.height;

      return { x, y };
    },
  };
};

const normalize = (image) => {
  image = ensureRgbMat(image);

  const { width, height, numChannels } = image;
  const hwc = image.getData({ channelsFirst: false });
  const output = new Float32Array(3 * height * width);

  // if input is integer in [0,255], convert to float32 in [0,1]
  const isUint8 = hwc.some((x) => x % 1 !== 0);
  const normFactor = isUint8 ? 1 : 255.0;

  // rgb
  const mean = [0.48145466, 0.4578275, 0.40821073];
  const std = [0.26862954, 0.26130258, 0.27577711];

  for (let i = 0; i < hwc.length; i += numChannels) {
    output[i + 0] = (hwc[i + 0] / normFactor - mean[0]) / std[0];
    output[i + 1] = (hwc[i + 1] / normFactor - mean[1]) / std[1];
    output[i + 2] = (hwc[i + 2] / normFactor - mean[2]) / std[2];
  }
  return Mat({ data: output, width, height, format: "RGB" });
};

const copyMakeBorder = (
  image,
  { top = 0, bottom = 0, left = 0, right = 0 }
) => {
  image = ensureRgbMat(image);
  const srcWidth = image.width;
  const srcHeight = image.height;
  const dstWidth = srcWidth + left + right;
  const dstHeight = srcHeight + top + bottom;
  const data = image.getData({ channelsFirst: false });

  const ArrayType = data.constructor;
  let output = new ArrayType(dstWidth * dstHeight * image.numChannels);

  for (var row = 0; row < srcHeight; row++) {
    for (var col = 0; col < srcWidth; col++) {
      const srcIdx = (row * srcWidth + col) * image.numChannels;
      const dstIdx =
        ((top + row) * dstWidth + (left + col)) * image.numChannels;
      output[dstIdx] = data[srcIdx];
      output[dstIdx + 1] = data[srcIdx + 1];
      output[dstIdx + 2] = data[srcIdx + 2];
      if (image.numChannels === 4) {
        output[dstIdx + 3] = data[srcIdx + 3];
      }
    }
  }

  return Mat({
    data: output,
    width: dstWidth,
    height: dstHeight,
    format: image.format,
  });
};

const resize = (image, { width: newWidth, height: newHeight }) => {
  image = ensureRgbMat(image);
  const { numChannels, width, height } = image;
  const arr = image.getData({ channelsFirst: false });

  const ArrayType = arr.constructor;
  const newArr = new ArrayType(newWidth * newHeight * numChannels);

  const xRatio = width / newWidth;
  const yRatio = height / newHeight;

  for (let y = 0; y < newHeight; y++) {
    const newY = Math.floor(y * yRatio);
    for (let x = 0; x < newWidth; x++) {
      const newX = Math.floor(x * xRatio);

      for (let c = 0; c < numChannels; c++) {
        const newIdx = (y * newWidth + x) * numChannels + c;
        const originalIdx = (newY * width + newX) * numChannels + c;

        newArr[newIdx] = arr[originalIdx];
      }
    }
  }

  return Mat({
    data: newArr,
    width: newWidth,
    height: newHeight,
    format: image.format,
  });
};

export { centerCrop, crop, normalize, copyMakeBorder, resize };
