"""
Setup script for Storage Forecasting Production System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="storage-forecasting",
    version="1.0.0",
    author="Storage Forecasting Team",
    description="A comprehensive system for disk storage monitoring and forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/storage-forecasting",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: System Administrators",
        "Topic :: System :: Monitoring",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "storage-forecast=src.core.main:main",
        ],
    },
)