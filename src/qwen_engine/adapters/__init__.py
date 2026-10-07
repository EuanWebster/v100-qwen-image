"""Adapter subpackage.

``base`` is CPU-safe (stdlib + engine types only). The only module permitted
to import torch / diffusers / transformers would be
``adapters/qwen21_diffusers.py`` — deliberately absent at M1 (plan R1).
"""
