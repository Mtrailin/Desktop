from setuptools import setup, find_packages

setup(
    name="method_validator",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'typing',
        'inspect',
        'functools',
        'logging'
    ],
    python_requires='>=3.7',
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool for validating method implementations and enforcing type hints",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
