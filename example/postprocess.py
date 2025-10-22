import numpy as np

def postprocess(onnx_output, class_labels, model_output_names, context=None):
    """
    General postprocessing function for ONNX runtime outputs.

    Parameters
    ----------
    onnx_output : dict[str, np.ndarray]
        Dictionary of NumPy arrays returned by ONNX inference.
    class_labels : list[str] or None
        Optional list of class labels. If provided, softmax outputs can be mapped to class names.
    model_output_names : list[str]
        Output tensor names from the ONNX model. Usually one, but multi-output models are allowed.
    context : dict (optional)
        Extra runtime metadata, for example:
            - return_argmax_only (bool)
            - output_activation ("softmax", "sigmoid", None)
            - top_k (int)
            - any other model-specific postprocessing needs

    Returns
    -------
    dict
        Clean, JSON-serializable inference result.
    """

    results = {}
    main_output_name = model_output_names[0]
    raw_output = onnx_output[main_output_name]

    # ----------------------------------------------------
    # DEFAULT BEHAVIOR: Apply softmax for classification
    # ----------------------------------------------------
    # (This preserves your MNIST logic but in a cleaner form)
    def softmax(x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

    probabilities = softmax(raw_output)

    # ----------------------------------------------------
    # OPTIONAL: Argmax classification
    # ----------------------------------------------------
    if class_labels:
        # Class names provided — return highest-probability class
        class_index = int(np.argmax(probabilities, axis=-1))
        results["predicted_index"] = class_index
        results["predicted_label"] = class_labels[class_index]

    # ----------------------------------------------------
    # ALWAYS: return full probability array (for debugging or UI)
    # ----------------------------------------------------
    results["probabilities"] = probabilities.tolist()

    return results