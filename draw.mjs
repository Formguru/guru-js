export function drawLine(ctx, a, b, options = {}) {
  var {
    color = "red",
    width = 5,
    alpha = 0.8,
    isDashed = false,
    lineDash = [3, 3],
  } = options;

  ctx.save();
  ctx.lineWidth = width;
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  if (isDashed) {
    ctx.setLineDash(lineDash);
  }
  ctx.beginPath();

  ctx.moveTo(a.position.x, a.position.y);
  ctx.lineTo(b.position.x, b.position.y);
  ctx.stroke();
  ctx.restore();
}

/**
 *
 * @param {canvas.context} ctx
 * @param {Array} points
 */
export function drawPath(ctx, points, options = {}) {
  const { color = "red", alpha = 1.0 } = options;

  if (points.length < 3) {
    return;
  }

  ctx.save();
  ctx.lineWidth = 4;
  ctx.strokeStyle = color;
  ctx.globalAlpha = alpha;
  ctx.beginPath();

  // Smooth path courtesy of https://stackoverflow.com/a/7058606/895769
  ctx.moveTo(points[0].x, points[0].y);
  for (var i = 1; i < points.length - 2; i++) {
    var xc = (points[i].x + points[i + 1].x) / 2;
    var yc = (points[i].y + points[i + 1].y) / 2;
    ctx.quadraticCurveTo(points[i].x, points[i].y, xc, yc);
  }
  // curve through the last two points
  ctx.quadraticCurveTo(
    points[i].x,
    points[i].y,
    points[i + 1].x,
    points[i + 1].y
  );

  ctx.stroke();
  ctx.restore();
}

function calcAngleBetween(v1, v2) {
  const dotProduct = (a, c) => a.x * c.x + a.y * c.y;
  const magnitude = (a) => Math.sqrt(a.x ** 2 + a.y ** 2);
  const cosTheta = dotProduct(v1, v2) / (magnitude(v1) * magnitude(v2));
  // floating point division could return a number just outside [-1, 1], e.g.,
  // 1.0000001 instead of 1, so we must clamp it
  return Math.acos(clamp(cosTheta, -1, 1));
}

/**
 * Return the angle ∠ABC in degrees
 */
export function getAngle(a, b, c) {
  const radiansToDegrees = (x) => {
    return (180 / Math.PI) * x;
  };

  const v1 = { x: a.x - b.x, y: b.y - a.y };
  const v2 = { x: c.x - b.x, y: b.y - c.y };
  return Math.round(radiansToDegrees(calcAngleBetween(v1, v2)));
}

/**
 * Label the angle between the angle ∠ABC in degrees
 * @param {canvas.context} ctx
 * @param {keypoint} a
 * @param {keypoint} b
 * @param {keypoint} c
 */
export function drawAngle(ctx, a, b, c, options = {}) {
  const {
    radius = 40,
    color = "white",
    alpha = 1.0,
    includeTextLabel = true,
    backgroundColor = MUI_PRIMARY,
  } = options;

  const getAngleFromPositiveXAxis = (p) => {
    return ensurePositive(
      Math.atan2(b.position.y - p.position.y, p.position.x - b.position.x)
    );
  };

  const toClockwiseAngle = (theta) => {
    // HTML canvas measures degrees *clockwise* from the positive x-axis
    return 2 * Math.PI - theta;
  };

  const ensurePositive = (theta) => {
    return theta < 0 ? theta + 2 * Math.PI : theta;
  };

  ctx.save();
  ctx.lineWidth = 3;
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.beginPath();

  const theta_a = getAngleFromPositiveXAxis(a);
  const theta_c = getAngleFromPositiveXAxis(c);
  const smaller = Math.min(
    toClockwiseAngle(theta_a),
    toClockwiseAngle(theta_c)
  );
  const bigger = Math.max(toClockwiseAngle(theta_a), toClockwiseAngle(theta_c));
  ctx.arc(
    b.position.x,
    b.position.y,
    radius,
    smaller,
    bigger,
    bigger - smaller <= Math.PI ? false : true
  );
  ctx.stroke();
  ctx.closePath();

  const angleInDegrees = getAngle(a.position, b.position, c.position);
  if (includeTextLabel) {
    const angleDirection =
      angleInDegrees > 0 && a.position.y > c.position.y ? -1 : 1;
    const prefix = options["angleLabel"] || "";
    ctx.font = "18px Roboto";
    const text = `${prefix}${(angleDirection * angleInDegrees).toString()}°`;
    const minDiameter = ctx.measureText("000°").width;
    const x = (a.position.x + b.position.x) / 2 - 5;
    const y = b.position.y + 20;
    inscribeTextInCircle(ctx, x, y, text, {
      alpha: 0.7,
      backgroundColor,
      minDiameter,
    });
  }

  ctx.restore();
  return angleInDegrees;
}

export function inscribeTextInCircle(ctx, x, y, text, options = {}) {
  const { minDiameter, backgroundColor = MUI_PRIMARY, alpha = 1.0 } = options;
  ctx.font = "18px Roboto";
  const diameter = Math.max(minDiameter, ctx.measureText(text).width);
  x = clamp(x, diameter / 2, ctx.canvas.width - diameter / 2);
  y = clamp(y, diameter / 2, ctx.canvas.height - diameter / 2);
  drawCircle(ctx, x, y, diameter / 2, { color: backgroundColor, alpha: alpha });

  ctx.fillText(text, x, y);
  ctx.restore();
  return { x: x, y: y, radius: diameter / 2 };
}

export function drawCircle(ctx, x, y, radius = 5, options = {}) {
  let { alpha = ".5", isFilled = true, lineWidth = 1, color } = options;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, 2 * Math.PI);
  if (isFilled) {
    ctx.fillStyle = color;
    ctx.fill();
  } else {
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = color;
    ctx.stroke();
  }
  ctx.restore();
}

export function drawRect(ctx, topLeftPos, bottomRightPos, options = {}) {
  let { alpha = 0.5, borderColor = null, borderWidth = 5, color } = options;
  ctx.save();
  ctx.globalAlpha = alpha;

  let width = bottomRightPos.x - topLeftPos.x;
  let height = bottomRightPos.y - topLeftPos.y;
  ctx.lineWidth = borderWidth;

  if (typeof color !== "undefined") {
    ctx.fillStyle = color;
    ctx.fillRect(topLeftPos.x, topLeftPos.y, width, height);
  } else {
    borderColor = borderColor || color || "black";
    ctx.strokeStyle = borderColor;
    ctx.strokeRect(topLeftPos.x, topLeftPos.y, width, height);
  }
  ctx.restore();
}

export const drawText = (ctx, text, x, y, maxWidth, options = {}) => {
  const {
    fontSize = 20,
    padding = 4,
    background = null,
    borderColor = null,
    borderWidth = 1,
    color = "#FFF",
    alpha = 1.0,
    textAlpha = null,
  } = options;

  const _textAlpha = textAlpha || alpha;
  // Performance optimization: don't do these calculations for invisible text
  if (_textAlpha === 0) {
    return;
  }

  ctx.globalAlpha = alpha;
  const font = `${Math.round(fontSize)}px Arial`;
  ctx.font = font;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  var words = text.split(" ");
  var lines = [];
  var currentLine = "";
  for (var n = 0; n < words.length; n++) {
    var testLine = currentLine + words[n] + " ";
    var testWidth = ctx.measureText(testLine).width;
    if (testWidth + x > maxWidth && n > 0) {
      lines.push(currentLine.trim());
      currentLine = words[n] + " ";
    } else {
      currentLine = testLine;
    }
  }
  lines.push(currentLine.trim());

  const labelWidth = Math.max(
    ...lines.map((line) => ctx.measureText(line).width)
  );
  const rectWidth = labelWidth + padding * 2;
  const rectHeight = fontSize * lines.length + padding * 2;
  if (background) {
    ctx.fillStyle = background;
    ctx.fillRect(x, y, rectWidth, rectHeight);
  }
  if (borderColor) {
    drawRect(
      ctx,
      { x: x, y: y },
      { x: x + rectWidth, y: y + rectHeight },
      { borderColor: borderColor, borderWidth: borderWidth }
    );
  }

  ctx.globalAlpha = _textAlpha;
  lines.forEach((line) => {
    var lineNumber = lines.indexOf(line) + 1;
    ctx.fillStyle = color;
    var lineHeight = padding + fontSize / 2 + fontSize / 16;
    ctx.fillText(
      line,
      x + labelWidth / 2 + padding,
      y + lineHeight * lineNumber
    );
  });
  return { width: rectWidth, height: rectHeight };
};

export function drawTriangle(ctx, a, b, c, options = {}) {
  let { alpha = 1.0, borderColor = null, borderWidth = 5, backgroundColor = null } = options;
  ctx.save();
  ctx.globalAlpha = alpha;

  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.lineTo(c.x, c.y);
  ctx.closePath();

  if (backgroundColor) {
    ctx.fillStyle = backgroundColor; // You can use any color you like
    ctx.fill();
  }
  else if (borderColor) {
    ctx.lineWidth = borderWidth;
    ctx.strokeStyle = borderColor;
    ctx.stroke();
  }

  ctx.restore();
}

export const drawCornerWatermark = (ctx, text, options = {}) => {
  const {
    fontSize = 20,
    padding = 4,
    background = "#F808",
    color = "#FFF",
    alpha = 1.0,
  } = options;

  // Performance optimization: don't do these calculations for invisible text
  if (alpha === 0) {
    return;
  }

  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.font = `${fontSize}px Roboto`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  const textWidth = ctx.measureText(text).width;
  const width = ctx.canvas.width;
  const height = ctx.canvas.height;
  const x = width - textWidth - padding;
  const y = height - fontSize - padding;

  ctx.fillStyle = background;
  ctx.fillRect(x, y, textWidth + padding * 2, fontSize + padding * 2);
  ctx.fillStyle = color;
  ctx.fillText(text, x + textWidth / 2 + padding, y + fontSize / 2 + padding);
  ctx.restore();
};

export const interpolateColor = (color1, color2, ratioFade) => {
  function hexToRgb(_hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(_hex);
    return result
      ? {
          r: parseInt(result[1], 16),
          g: parseInt(result[2], 16),
          b: parseInt(result[3], 16),
        }
      : null;
  }

  function hex(c) {
    const s = "0123456789abcdef";
    let i = parseInt(c);
    if (i === 0 || isNaN(c)) return "00";
    i = Math.round(Math.min(Math.max(0, i), 255));
    return s.charAt((i - (i % 16)) / 16) + s.charAt(i % 16);
  }

  function rgbToHex(rgb) {
    return "#" + hex(rgb[0]) + hex(rgb[1]) + hex(rgb[2]);
  }

  const startRGB = hexToRgb(color1);
  const endRGB = hexToRgb(color2);

  let diffRed = endRGB.r - startRGB.r;
  let diffGreen = endRGB.g - startRGB.g;
  let diffBlue = endRGB.b - startRGB.b;

  diffRed = diffRed * ratioFade + startRGB.r;
  diffGreen = diffGreen * ratioFade + startRGB.g;
  diffBlue = diffBlue * ratioFade + startRGB.b;

  return rgbToHex([
    Math.round(diffRed),
    Math.round(diffGreen),
    Math.round(diffBlue),
  ]);
};

function clamp(x, lo, hi) {
  return Math.min(Math.max(x, lo), hi);
}

export const MUI_PRIMARY = "#9c27b0";
export const MUI_SECONDARY = "#f44336";
