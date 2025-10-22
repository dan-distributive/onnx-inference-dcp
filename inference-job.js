/**
 * @file    dcp-inference.js
 * @brief   Demonstrates batch inference using MNIST:
 *          - Python preprocessing
 *          - ONNX runtime inference
 *          - Python postprocessing
 * 
 * @author  Ryan Saweczko, ryansaweczko@distributive.network
 * @author  Dan Desjardins, dan@distributive.network
 * @date    October 22nd, 2023
 * @version 1.0
 * @license MIT
 */
'use strict';

async function main() {
    const compute = require('dcp/compute');
    const fs = require('fs');
    const path = require('path');

    // -------------------------
    // MODEL INFO
    // -------------------------
    // This object contains metadata for your inference job. 
    // Most machine learning models (TensorFlow, PyTorch, Keras, etc.) can be converted to ONNX format.
    // See https://onnx.ai/ for documentation on converting models to ONNX.
    const modelInfo = {
        name: 'MNIST on DCP!',                              // name of your inference job or model
        labels: ["0","1","2","3","4","5","6","7","8","9"],  // for MNIST
        model: './example/MNIST.onnx',                      // relative path to your .onnx model
        preprocess: './example/preprocess.py',              // path to your python preprocessing file
        postprocess: './example/postprocess.py',            // path to your python postprocessing file
        packages: ['numpy', 'opencv-python']                // python packages required by your python pre and post processing scripts
    };

    // -------------------------
    // INPUT SET
    // -------------------------
    const inputDir  = './example/input';
    const batchSize = 3;

    // Read all .png files in the directory
    const inputFilenames = fs.readdirSync(inputDir)
        .filter(f => f.toLowerCase().endsWith('.png'));

    const inputSet = [];
    let currentBatch = {};

    for (const file of inputFilenames) {
        const b64 = fs.readFileSync(path.join(inputDir, file)).toString('base64');
        currentBatch[path.parse(file).base] = b64;

        // Push batch when full
        if (Object.keys(currentBatch).length === batchSize) {
            inputSet.push({ b64Data: currentBatch });
            currentBatch = {};
        }
    }

    // Push any remaining images
    if (Object.keys(currentBatch).length) inputSet.push({ b64Data: currentBatch });

    // Push any leftovers (not filling a full batch)
    if (Object.keys(currentBatch).length > 0) {
        inputSet.push({ b64Data: currentBatch });
    }

    // -------------------------
    // WORK FUNCTION
    // -------------------------
    const { inferenceFunction } = require('./inference-function');   // inferenceFunction(input, classLabels, model, preprocess, postprocess, packages, metadata)

    // -------------------------
    // ARGUMENTS
    // -------------------------
    const args = [
        modelInfo.labels,                                           // model class labels
        fs.readFileSync(modelInfo.model).toString('base64'),        // base64-encoded onnx model
        fs.readFileSync(modelInfo.preprocess).toString('utf-8'),    // utf-8-encoded python preprocessing file
        fs.readFileSync(modelInfo.postprocess).toString('utf-8'),   // utf-8-encoded python postprocessing file
        modelInfo.packages,                                         // list of pyodide packages to fetch from the DCP package manager (Ex: numpy, scikit-learn)
        { webgpu: true },                                           // metadata object 
    ];

    // -------------------------
    // DCP JOB
    // -------------------------
    const job = compute.for(inputSet, inferenceFunction, args);
    
    // DCP Job config
    job.public.name = `${modelInfo.name}`;
    job.computeGroups = [{ joinKey: 'demo', joinSecret: 'dcp' }];
    job.requires([
        'onnxruntime-dcp/dcp-wasm.js', 
        'onnxruntime-dcp/dcp-ort.js', 
        'pyodide-core/pyodide-core.js'
    ]);
    job.requirements.environment = { webgpu: true }

    // DCP events
    job.on('readystatechange', (s) => console.log(`State: ${s}`))
    job.on('accepted', () => console.log(`Job accepted with id: ${job.id}`));
    job.on('result', (r) => console.log(JSON.stringify(r.result, null, 2)));
    job.on('error', (e) => console.error(e));
    job.on('console', (c) => console.dir(c, {depth:Infinity}))

    // -------------------------
    // EXECUTE JOB
    // -------------------------
    const resultSet = await job.exec();

    // -------------------------
    // PROCESS & DISPLAY RESULTS
    // -------------------------
    console.log('\nAll inference results:\n');
    // Print headers
    console.log(`Input\t\tLabel\tIndex\tConfidence`);
    console.log(`------------------------------------------------`);
    for (const result of resultSet) {
        for (const [key, value] of Object.entries(result)) {
            // Handle probabilities (support batch size > 1)
            const batchProbs = value.probabilities || [];
            const outputArray = Array.isArray(batchProbs[0]) ? batchProbs[0] : batchProbs;

            // Predicted index
            const predIndex = value.predicted_index !== undefined
                ? value.predicted_index
                : outputArray.indexOf(Math.max(...outputArray));

            // Predicted label (optional)
            const predLabel = value.predicted_label !== undefined
                ? `${value.predicted_label}`
                : '';

            // Confidence percentage
            const confidence = (outputArray[predIndex] * 100).toPrecision(3);
            
            // Print results
            console.log(`${key}\t\x1b[33m${predLabel}\t\x1b[0m${predIndex}\t${confidence}%`);
        }
    }
    console.log();

}

// -------------------------
// INIT DCP AND RUN
// -------------------------
require('dcp-client').init().then(main);