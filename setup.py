"""
Setup script for Valerie Visual ASR package.
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="valerie-visual-asr",
    version="0.1.0",
    author="Valerie Research Team",
    author_email="contact@valerie-asr.com",
    description="Enhanced Visual ASR Language Model with Conformer Architecture and Knowledge Distillation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/valerie-team/valerie-visual-asr",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "cloud": [
            "boto3>=1.24.0",
            "google-cloud-storage>=2.5.0",
        ],
        "visualization": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "valerie-train=scripts.train:main",
            "valerie-evaluate=scripts.evaluate:main",
            "valerie-inference=scripts.inference:main",
            "valerie-prepare-data=scripts.prepare_data:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    zip_safe=False,
    keywords=[
        "visual speech recognition",
        "lip reading",
        "conformer",
        "knowledge distillation",
        "pytorch",
        "deep learning",
        "computer vision",
        "natural language processing",
    ],
    project_urls={
        "Bug Reports": "https://github.com/valerie-team/valerie-visual-asr/issues",
        "Source": "https://github.com/valerie-team/valerie-visual-asr",
        "Documentation": "https://valerie-visual-asr.readthedocs.io/",
    },
)