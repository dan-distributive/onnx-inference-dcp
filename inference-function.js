/*
 *  @file   inference-function.js
 *
 *  @author Erin Peterson, erin@distributive.network
 *  @author Mehedi Arefin, mehedi@distributive.network
 *  @author Dan Desjardins, dan@distributive.network
 *  @date   October 22nd, 2025
 */
'use strict';

/**
 * Work function that is passed to DCP Worker
 * @async
 * @function
 * @name inferenceFunction
 * @param { object } sliceData
 * @param { object } classLabels
 * @param { object } metadata
 */
async function inferenceFunction(sliceData, classLabels, modelB64, preprocessStr, postprocessStr, pythonPackages, metadata) {
    // ======================================================
	// 1. WORKER & RUNTIME SETUP
	// ======================================================
	progress(0);
	require('dcp-wasm.js');

	// ======================================================
	// 2. MODEL & ONNX SESSION INITIALIZATION
	// ======================================================
	const model = b64ToArrayBuffer(modelB64);
	let onnxInput      = {};
	let onnxOutput     = {};
	let batchedResults = {};

	progress(0.1);

	// Load ONNX Runtime once per worker
	if (!globalThis.ort) {
		globalThis.ort = require('dcp-ort.js');
	}
	ort.env.wasm.simd = true;

	// Create or reuse ONNX session
	if (!globalThis.session) {
		globalThis.session = await ort.InferenceSession.create(model, {
			executionProviders: [metadata['webgpu'] ? 'webgpu' : 'wasm'],
			graphOptimizationLevel: 'all'
		});
	}

	// Model metadata
	const modelInputNames  = session.inputNames;
	const modelOutputNames = session.outputNames;

	progress(0.2);

	// ======================================================
	// 3. UTILITY FUNCTIONS
	// ======================================================
	/**
	 * Convert B64 to Array Buffer in DCP work function
	 * @function
	 * @name b64ToArrayBuffer
	 * @param { string } base64
	 * @returns { Uint8Array.buffer }
	 */
	function b64ToArrayBuffer(base64) {
		const binary = atob(base64);
		const len    = binary.length;
		const bytes  = new Uint8Array(len);

		for (let i = 0; i < len; i++) {
			bytes[i] = binary.charCodeAt(i);
		}

		return bytes.buffer;
	};

	/**
	 * Map inference results from a list of key-value pairs into an object in DCP work function
	 * @function
	 * @name mapToObj
	 * @param  { Map<string, string>} m - Map which is to be converted to an object
	 * @returns { object } obj
	 */
	function mapToObj(m) {
		const obj = Object.fromEntries(m);
		for (const key of Object.keys(obj)) {
			if (obj[ key ].constructor.name == 'Map') {
				obj[ key ] = mapToObj(obj[ key ]);
			}
		}
		return obj;
	};

	/**
	 * Performs the inference on the onnx input tensors
	 * @async
	 * @function
	 * @name runInference
	 * @param { object } onnxInput
	 * @returns { Promise<object> } infOut
	 */
	async function runInference(onnxInput) {
		const onnxOutput   = await session.run(onnxInput);
		return onnxOutput;
	};


	// ======================================================
	// 4. PYTHON INFERENCE PIPELINE (PRE → ONNX → POST)
	// ======================================================
	/**
	 * Python preprocessing, inference, and postprocessing pipeline
	 * @async
	 * @function
	 * @name pythonInferencePipeline
	 * @param { Array<string> } pythonPackages
	 * @param { object } sliceData
	 * @param { object } classLabels
	 * @param { object } metadata
	 * @returns { object } batchedResults
	 */
	async function pythonInferencePipeline(sliceData, classLabels, pythonPackages, metadata) {
		// ------------------------
		// 4.1 Initialize Pyodide
		// ------------------------
		if (!globalThis.pyodideCore) {
			globalThis.pyodideCore = require('pyodide-core.js');
		}
		if (!globalThis.pyodide) {
			globalThis.pyodide = await pyodideCore.pyodideInit();
		}
		await pyodideCore.loadPackage(pythonPackages);

		// Inject python pre and post processing functions
		globalThis.preprocessStr  = preprocessStr;
		globalThis.postprocessStr = postprocessStr;
		try {
			pyodide.runPython(`
				import js
				exec(js.globalThis.preprocessStr)
				preprocessFunction = preprocess
				exec(js.globalThis.postprocessStr)
				postprocessFunction = postprocess
			`);
		} catch(error) {
			const stack = error.message.split('\n');
			const errorMsg = stack[stack.length - 2];
      		console.log('error', error)
			return {
				'code': 'pyodide',
				'msg' : errorMsg,
				'input': metadata['inputID']
			};
		}

		// ------------------------
		// 4.2 Core Inference Loop
		// ------------------------
		// loops when client-side slice batch size > 1
		const entries = Object.entries(sliceData.b64Data);
		for (let i = 0; i < entries.length; i++) {
			const [key, value] = entries[i];
			
			progress(0.2 + i * (0.8 / entries.length));

			// Decode slice data input
			metadata['inputID'] = key;
			const abInput = b64ToArrayBuffer(value);
			
			//
			// 4.2.1 Python preprocessing
			//---------------------------
			// Push slice input data into pyodide
			pyodide.globals.set('preprocessArgs', [abInput, modelInputNames]);
			
			// Run preprocessing entirely in Python
			try {
				npArrays = pyodide.runPython(`
					import numpy as np
					preprocessArgs = preprocessArgs.to_py()
					bytes = preprocessArgs[0].tobytes()
					model_input_names = np.array(preprocessArgs[1])
					
					np_arrays = preprocessFunction(bytes, model_input_names)
					
					for(key, array) in np_arrays.items():
						np_arrays[key] = np.ascontiguousarray(array, dtype=array.dtype)
					np_arrays
        		`);
			} catch(error) {
				const stack = error.message.split('\n');
				const errorMsg = stack[stack.length - 2];
				return {
					'code': 'preprocess',
					'msg' : errorMsg,
					'input': metadata['inputID']
				};
			} finally {
				pyodide.globals.pop('preprocessArgs')  // cleanup
			};

			// Convert numpy arrays to tensors
			for (const key of npArrays.keys()) {
				const value  = npArrays.get(key);
				onnxInput[ key ] = new ort.Tensor(
					value.dtype.name,
					value.getBuffer().data,
					value.shape.toJs());
			};

			// 4.2.2 Run Inference
			//---------------------------
			try {
				// Call ONNX runtime with prepared feeds
				onnxOutput = await runInference(onnxInput);
				onnxInput  = {};
			} catch(error) {
				return {
					'code': 'inference',
					'msg' : error.message,
					'input': metadata['inputID']
				};
			}
			
			// 4.2.3 Python postprocessing
			//----------------------------
			// Ensure each ONNX tensor exposes its underlying ArrayBuffer for Python
			for (const key of Object.keys(onnxOutput)) {
				const value       = onnxOutput[key];
				value.data_buffer = value.data.buffer;
				onnxOutput[key]   = value;
			}

			// Push ONNX output and metadata into Pyodide
			pyodide.globals.set('postprocessArgs', [onnxOutput, classLabels, modelOutputNames]);

			try {
				// Convert ONNX outputs to contiguous NumPy arrays and run Python postprocessing
				pyResult = pyodide.runPython(`
					import numpy as np

					# Unpack ONNX outputs and other arguments from JS
					onnx_output = postprocessArgs.to_py()[0]
					class_labels = postprocessArgs.to_py()[1]
					model_output_names = postprocessArgs.to_py()[2]

					# Convert each tensor to contiguous NumPy array
					for key, value in onnx_output.items():
						dims = value.dims.to_py()
						dtypeStr = str(value.type)
						buffer = value.data_buffer.to_py()
						onnx_output[key] = np.frombuffer(buffer.tobytes(), dtype=dtypeStr).reshape(dims)

					# Run user-defined postprocessing function
					postprocessFunction(onnx_output, class_labels, model_output_names)
				`)
			} catch(error) {
				const stack    = error.message.split('\n');
				const errorMsg = stack[stack.length - 2];
				return {
					'code': 'postprocess',
					'msg' : errorMsg,
					'input': metadata['inputID']
				};
			} finally {
				pyodide.globals.pop('postprocessArgs')  // cleanup Python globals
			}

			// 4.2.4 Convert py to js result
			//----------------------------
			jsResult = mapToObj(pyResult.toJs());
			batchedResults[key] = jsResult;
		}
		// ------------------------
		// 4.3 Finish & Return
		// ------------------------
		return batchedResults;
	};

	// ======================================================
	// 5. EXECUTE PIPELINE AND RETURN RESULTS
	// ======================================================
	batchedResults = await pythonInferencePipeline(sliceData, classLabels, pythonPackages, metadata);
	progress(1);
	return batchedResults;
}

exports.inferenceFunction = inferenceFunction;