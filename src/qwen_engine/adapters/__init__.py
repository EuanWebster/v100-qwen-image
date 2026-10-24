"""Adapter subpackage.

``base`` is CPU-safe (stdlib + engine types only). The only modules permitted
to import torch / diffusers / transformers are ``supplied_qwen21`` and
``qwen21_diffusers``, and only inside their lazy load paths — plain imports of
either module stay ML-free (plan R1).
"""
