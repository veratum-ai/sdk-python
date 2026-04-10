"""Setup configuration for veratum-verify package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="veratum-verify",
    version="0.1.0",
    author="Veratum Inc.",
    author_email="info@veratum.ai",
    description="Independently verify Veratum compliance receipts — no account required",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/veratum-ai/veratum-v2",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Security :: Cryptography",
    ],
    python_requires=">=3.8",
    install_requires=[],  # Zero external dependencies
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "veratum-verify=veratum_verify.cli:main",
        ],
    },
    license="MIT",
    keywords="veratum compliance audit transparency merkle tree verification",
)
