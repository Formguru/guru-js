export const GURU_KEYPOINTS = [
  "nose",
  "left_eye",
  "right_eye",
  "left_ear",
  "right_ear",
  "left_shoulder",
  "right_shoulder",
  "left_elbow",
  "right_elbow",
  "left_wrist",
  "right_wrist",
  "left_hip",
  "right_hip",
  "left_knee",
  "right_knee",
  "left_ankle",
  "right_ankle",
  "left_heel",
  "right_heel",
  "left_toe",
  "right_toe",
];

export class Color {
  /**
   * A colour, represented as an RGB value. Valid values for each are >= 0 and <= 255.
   *
   * @param {number} r - The amount of red in the color.
   * @param {number} g - The amount of green in the color.
   * @param {number} b - The amount of blue in the color.
   */
  constructor(r, g, b) {
    this.r = r;
    this.g = g;
    this.b = b;
  }

  toHex() {
    return `#${this.r.toString(16).padStart(2, "0")}${this.g
      .toString(16)
      .padStart(2, "0")}${this.b.toString(16).padStart(2, "0")}`;
  }
}

export class Position {
  /**
   * The two-dimensional coordinates indicating the location of something.
   *
   * @param {number} x - The x coordinate of the position.
   * @param {number} y - The y coordinate of the position.
   * @param {number} confidence - The confidence Guru has of the accuracy of this position.
   *        0.0 implies no confidence, 1.0 implies complete confidence.
   */
  constructor(x, y, confidence) {
    this.x = x;
    this.y = y;
    this.confidence = confidence !== undefined ? confidence : 1.0;
  }

  interpolate(other, factor) {
    /**
     * Interpolates this position with another, with the difference weight by a given factor.
     *
     * @param {Position} other - The other Position to interpolate between.
     * @param {number} factor - The scaling weight to apply to the difference in location between the two positions.
     * @returns {Position} The Position interpolated between this one and other.
     */
    return new Position(
      this.x + (other.x - this.x) * factor,
      this.y + (other.y - this.y) * factor,
      (this.confidence + other.confidence) / 2
    );
  }

  toImageCoords(width, height) {
    /**
     * Converts this position to image coordinates.
     *
     * @param {number} width - The width of the image.
     * @param {number} height - The height of the image.
     * @returns {Tuple} A tuple of length 2. The first element is the x coordinate in the image, the second is the y coordinate.
     */
    return [Math.floor(this.x * width), Math.floor(this.y * height)];
  }
}

export class Box {
  /**
   * A two-dimensional box, often indicating the bounds of something.
   *
   * @param {Position} top_left - The top-left corner of the box.
   * @param {Position} bottom_right - The bottom-right corner of the box.
   */
  constructor(top_left, bottom_right) {
    this.topLeft = top_left;
    this.bottomRight = bottom_right;
  }
}

function snakeToLowerCamelCase(snake) {
  return snake.replace(/_([a-z])/g, function (match, letter) {
    return letter.toUpperCase();
  });
}

/**
 * Keypoint is an enumeration of the names of the keypoints which can be found on objects.
 */
export const Keypoint = Object.freeze(GURU_KEYPOINTS.reduce((keypointEnum, keypointName, index) => {
  const lowerCamelCase = snakeToLowerCamelCase(keypointName);
  keypointEnum[lowerCamelCase] = keypointName;
  return keypointEnum;
}, {}));

/**
 * An enumeration of the ways an object can be facing, relative to the camera.
 */
export const ObjectFacing = Object.freeze({
  /**
   * Object facing the left-side of the frame.
   */
  Left: "left",
  /**
   * Object facing the right-side of the frame.
   */
  Right: "right",
  /**
   * Object facing towards the camera.
   */
  Toward: "toward",
  /**
   * Object facing away from the camera.
   */
  Away: "away",
  /**
   * Object facing towards the top of the frame.
   */
  Up: "up",
  /**
   * Object facing towards the bottom of the frame.
   */
  Down: "down",
  /**
   * Object direction unknown.
   */
  Unknown: "unknown",
});