# guru.js -- Javascript Libraries for Computer Vision

## Overview

`guru-js` is a Javascript framework for using state-of-the-art vision models for specific computer vision tasks.

**❗ Important Note:** These libraries implement a common execution environment for [Guru Schemas](https://docs.getguru.ai/quickstart/guru-schema-intro) and are provided to users at runtime via the [Guru platform](https://www.getguru.ai/). They are not intended for direct use outside the Guru platform. This repository is primarily for documentation, issue tracking, and contributing to the library's development.

## Quickstart

To use these libraries, sign up in the [Guru Console](https://console.getguru.ai/) and create your first [Guru Schema](https://docs.getguru.ai/quickstart/guru-schema-intro). Check out the [Guru Docs]() and the [Guru Discord](https://discord.gg/tCTPVkSCas) for more info and support.

## Features

**Modular Design**: Combine advanced vision models in a few lines of standard Javascript.

**Analysis Helper Functions**: Analyzers provide easy-to-use libraries for domain-specific AI analysis for domains such as `MovementAnalyzer` for sports/fitness.

**Advanced Guru Models**: `guru.js` is pre-packaged with state-of-the-art models for tracking human movement and objects in real-time.

**Bring Your Own Model**: Import and leverage any ONNX model into a `guru-js` app .

**Customizable Visualization Tools**: [Visualize AI analysis results](https://www.loom.com/share/458a8cf435a64f01ba8fa86454d9f013) in real-time on any video.

**Deploy Anywhere**: Apps built with `guru.js` can be deployed on the cloud or mobile in [one-click](https://docs.getguru.ai/deploying/guru-api-intro).

## Core Libraries

- `stdlib.mjs`: Utility functions for analyzing outputs of vision models
- `core_types.mjs`: Defines fundamental types and structures used across the library, forming the backbone of the library's modular architecture.
- `draw.mjs`: Offers drawing and annotation tools for visualization purposes, enhancing the interpretability of computer vision outputs.
- `pose_estimation.mjs`: Implements pose estimation functionalities.
- `onnxruntime.mjs` and `onnx_model.mjs`: Facilitate the integration and use of ONNX (Open Neural Network Exchange) models, ensuring efficient model performance and advanced machine learning capabilities.
- `object_tracking.mjs`: Provides object tracking features, essential for tracking objects across video frames.
- `object_detection.mjs`: Implements object detection algorithms, a cornerstone for many computer vision applications.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions and support, please open an issue in the repository or contact the maintainers directly at support@getguru.ai or via our [Discord](https://discord.gg/tCTPVkSCas).
