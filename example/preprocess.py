import numpy as np
import cv2 as cv   # (optional; only needed for images)

def preprocess(bytes_input, model_input_names, context=None):
    """
    General preprocessing function for ONNX runtime.

    Parameters
    ----------
    bytes_input : bytes
        The raw input payload from JavaScript (image bytes, audio bytes, text bytes, etc.)
    model_input_names : list[str]
        The ONNX model's expected input tensor names. The returned dictionary must use these keys.
    context : dict (optional)
        Optional runtime metadata, for example:
            - tokenizer (for NLP)
            - image_size or scaling info
            - normalization constants
            - arbitrary preprocessing parameters

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary mapping ONNX input tensor names to contiguous NumPy arrays.
        Example: { "input_ids": ndarray, "attention_mask": ndarray }
    """

    onnx_input = {}

    # --- Example: IMAGE CASE (MNIST-style) ---
    arr = np.frombuffer(bytes_input, dtype=np.uint8)
    image = cv.imdecode(arr, cv.IMREAD_COLOR)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = image / 255.0
    image = cv.resize(image, (28, 28))
    image = np.reshape(image.astype(np.float32), (1, 1, 28, 28))

    # Use the FIRST model input name for now (multi-input supported later)
    model_input_name = model_input_names[0]
    onnx_input[model_input_name] = np.ascontiguousarray(image)

    return onnx_input