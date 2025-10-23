# Distributed ONNX Model Inference on DCP
This project demonstrates how to run batch machine learning inference on the Distributive Compute Platform (DCP) using:

* Python preprocessing (`preprocess.py`)
* ONNX runtime inference (`MNIST.onnx`)
* Python postprocessing (`postprocess.py`)
* Distributed workers (browser workers or native workers)

This example is intended to illustrate the complete end-to-end workflow for distributed inference. While this MNIST demo is small, the same structure can be applied to larger models and datasets.

> Note: This demo is slower than running inference locally due to network transfer and worker initialization overhead. The performance benefits of DCP become evident when preprocessing, inference, and postprocessing are computationally expensive and the input volume is large.

## Overview

This repository is designed as a tutorial rather than a production-ready system. The code is intended to be clear and easy to understand, allowing developers to modify it as needed. Many configuration values are hard-coded (model path, input folder, batch size, compute group, etc.), but they can be adapted or parameterized for more advanced use cases.

If you encounter any issues or have questions, you can reach the team via:

* Email: info@distributive.network
* Slack: [DCP Developers Slack](https://join.slack.com/t/dcp-devs/shared_invite/zt-56v87qj7-fkqZOXFUls8rNzO4mxHaIA)

## Prerequisites

* Node.js
* *dcp-client* library:
```
npm i dcp-client
```
* DCP keystore files in the home directory:
```
~/.dcp/id.keystore
~/.dcp/default.keystore
```
To obtain keystores, contact: dan@dcp.dev

## Running the Example

1. Launch the inference job:
```
node inference-job.js
```
2. Start browser-based DCP Workers:
* Open https://dcp.work/demo
* Enter join secret: `dcp`
* Click **Start**
3. As job sslices are processed, results will be displayed in the terminal:
```
Input      Label  Index  Confidence
-----------------------------------
60110.png   8      8      99.9%
```
Browser-based DCP Workers are sufficient for testing, but production-scale workflows can use native DCP Docker, Linux, or Windows Workers. More information: [DCP Workers](https://distributive.network/workers)

## Project Structure
```
.
├── inference-job.js        # Main job script
├── inference-function.js   # Worker-side function: preprocess → infer → postprocess
├── example/
│   ├── MNIST.onnx          # ONNX model
│   ├── preprocess.py
│   ├── postprocess.py
│   └── input/*.png         # Example images
└── package.json
```

## Configuration
The following parameters can be modified for customization:

| Parameter                    | Location                       | Description                                                                      |
| ---------------------------- | ------------------------------ | -------------------------------------------------------------------------------- |
| Input images                 | `example/input`                | Replace with your own inputs                                                     |
| Batch size                   | `batchSize`                    | Number of inputs to batch (`const batchSize`)                                    |
| Model and labels             | `modelInfo`                    | Swap ONNX model and class labels                                                 |
| Python preprocessing script  | `modelInfo.preprocess`         | Path to the preprocessing Python file                                            |
| Python postprocessing script | `modelInfo.postprocess`        | Path to the postprocessing Python file                                           |
| Required Python packages     | `modelInfo.packages`           | List of Pyodide packages for the worker to load (e.g., `numpy`, `opencv-python`) |
| Job name                     | `job.public.name`              | Name displayed in the DCP job list and in the workers                            |
| Compute group                | `job.computeGroups`            | Set joinkKey and joinSecret for the group into which your job will deploy        |
| Worker requirements          | `job.requires`                 | List of JavaScript libraries for the worker to load                              |
| Execution environment        | `job.requirements.environment` | Environment flags for the worker (e.g., `webgpu: true`)                          |
| Console logging              | `job.on('console', ...)`       | Optional: enable or disable logging of worker console messages in the client     |
| Result handling              | `job.on('result', ...)`        | Optional: dynamically receive and handle individual result events                |
| Error handling               | `job.on('error', ...)`         | Callback for handling errors from workers                                        |


To reduce console verbosity from workers, you can comment out or remove:
```
job.on('console', (c) => console.dir(c, { depth: Infinity }));
```
Other event listeners, such as `result`, can also be disabled if you prefer to process all results only after the job completes.

## Extending the Example

The same structure can be used for other types of models or workflows:
* Image classification or segmentation
* Audio or video analysis
* Natural Language Processing (NLP)
* Medical imaging pipelines
* etc, etc
The pattern remains:
```
preprocess → infer → postprocess → return results
```
