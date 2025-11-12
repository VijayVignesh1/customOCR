from setuptools import setup, find_packages
from pathlib import Path

def load_requirements():
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        return requirements_file.read_text().strip().split('\n')
    return []

setup(
    name="customocr",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=load_requirements(),
    author="Vijay Vignesh",
    description="A quick and easy pipeline for generating OCR dataset and finetuning a model",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/VijayVignesh1/customOCR",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)