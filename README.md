# Guru Computer Vision Javascript Libraries

## Overview

These Javascript libraries implement a common execution environment for [Guru Schemas](https://docs.getguru.ai/quickstart/guru-schema-intro), tailored specifically for advanced computer vision tasks such as object detection, pose estimation, and movement analysis.

**Important Note:** These libraries are built into the [Guru platform](https://www.getguru.ai/) and are provided to users at runtime. They are not intended for direct use outside the Guru platform. This repository is primarily for documentation, issue tracking, and contributing to the library's development.

You can sign up for Guru [here](https://console.getguru.ai/). Check out our [docs](https://docs.getguru.ai/introduction) and [Discord](https://discord.gg/tCTPVkSCas) for more info and support.

## Quickstart

Create your first schema in the [Guru Console](https://docs.getguru.ai/quickstart/guru-console-intro) to play with these libraries.

## Features

`guru-js` is a framework for combining and analyzing state-of-the-art vision models for specific computer vision tasks. This includes:

- **Advanced Pose Estimation, Object Detection, and Tracking**: State-of-the-art models for analyzing human movement and object behavior in real-time.
- **Analysis Helper Functions**: `Analyzers` like `MovementAnalyzer` that provide easy-to-use libraries for domain-specific analysis of model output like sports or retail.
- **Customizable Visualization Tools**: Drawing and annotation capabilities to visualize analysis results on images and videos.
- **Bring Your Own Model**: Import, leverage, and combine any ONNX model into one AI workflow.
- **Deploy Anywhere**: With the Guru Platform, any AI app built with these tools can be deployed on the cloud or mobile in [one-click](https://docs.getguru.ai/deploying/guru-api-intro).

## Core Libraries

- `stdlib.mjs`: Fundamental utility classes and functions for basic operations like frame and object handling in computer vision tasks.
- `pose_estimation.mjs`: Implements pose estimation functionalities, crucial for human movement analysis in images and videos.
- `onnxruntime.mjs` and onnx_model.mjs: Facilitate the integration and use of ONNX (Open Neural Network Exchange) models, ensuring efficient model performance and advanced machine learning capabilities.
- `object_tracking.mjs`: Provides object tracking features, essential for tracking objects across video frames.
- `object_detection.mjs`: Implements object detection algorithms, a cornerstone for many computer vision applications.
- `core_types.mjs`: Defines fundamental types and structures used across the library, forming the backbone of the library's modular architecture.
- `draw.mjs`: Offers drawing and annotation tools for visualization purposes, enhancing the interpretability of computer vision outputs.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions and support, please open an issue in the repository or contact the maintainers directly at support@getguru.ai or via our [Discord](https://discord.gg/tCTPVkSCas).
