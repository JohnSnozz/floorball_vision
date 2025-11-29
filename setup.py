"""Setup script for Floorball Vision."""

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='floorball_vision',
    version='0.1.0',
    description='Computer Vision System for Floorball Analysis',
    author='Jonas',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'fb-download=src.data.youtube_downloader:main',
            'fb-screenshot=src.data.screenshot_extractor:main',
            'fb-batch=src.data.batch_creator:main',
            'fb-split=src.data.data_splitter:main',
            'fb-train=src.training.train:main',
        ],
    },
)
