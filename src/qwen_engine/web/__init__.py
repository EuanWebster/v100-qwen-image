"""W2 local web front-end: same-origin status/plan API + static server.

ML-free by construction: nothing here imports torch/diffusers/transformers
or loads a weight file. The server only plans (``core.plan``) and validates
``GenerationConfig``; generation is W3 and stays unimplemented. See
``qwen_engine.web.server`` (transport) and ``qwen_engine.web.api`` (pure
contract logic).
"""
