"""Compatibility shim. Project metadata lives in :file:`pyproject.toml`.

This file remains so that workflows that still invoke ``setup.py`` keep
working. Edit ``pyproject.toml`` instead of adding fields here.
"""
from setuptools import setup

setup()
