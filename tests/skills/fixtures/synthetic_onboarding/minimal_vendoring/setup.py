from pathlib import Path

from setuptools import setup

version = Path("VERSION").read_text().strip()
setup(name="synthetic-vendored-audio", version=version)
