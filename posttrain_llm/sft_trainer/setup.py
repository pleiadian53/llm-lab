"""Setup script for sft_trainer package."""

from setuptools import setup, find_packages

setup(
    name="sft_trainer",
    version="0.1.0",
    description="Supervised Fine-Tuning with PEFT Support",
    author="pleiadian53",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "datasets>=2.14.0",
        "peft>=0.10.0",
        "trl>=0.8.0",
        "accelerate>=0.27.0",
        "bitsandbytes>=0.42.0",
        "pandas>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
