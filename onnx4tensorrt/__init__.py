"""
    Although this package provide the scripts and implementations to
    support ONNX conversion and deploy on TensorRT.

    It kinda breaks the "DRY" principle" but the the model and backbone need some tweak
    and more attention than the pure pytorch counterpart. Since issue related to the
    particular contexts of ONNX and TensorRT may be raised in the future, it's safer
    to separate the two versions for the moment.
"""