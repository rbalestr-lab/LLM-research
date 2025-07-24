from setuptools import setup, find_packages

setup(
    name="spurious_corr",
    version="0.1.0",
    description="A library for injecting spurious correlations into text data.",
    author="Pradyut Sekhsaria, Marcel Mateos Salles, Hai Huang, Randall Balestriero",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    install_requires=[
        "datasets>=2.0.0",
        "termcolor>=1.1.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    extras_require={"dev": ["pytest", "pytest-cov", "black"]},
)
