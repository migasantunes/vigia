"""Export and serving (plan Phase 6).

``export`` needs the ``train`` extra (PyTorch -> ONNX); ``runtime``, ``service`` and
``replay`` need the ``serve`` extra (ONNX Runtime, FastAPI) and never import PyTorch.
"""
