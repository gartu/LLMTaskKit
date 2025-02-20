from setuptools import setup, find_packages
import os

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="LLMTaskKit",
    version="1.0.0",
    author="Herzog Robin",
    author_email="llmtaskkit@gartu.xyz",
    description="A Python bookstore to simplify the definition, execution and chaining of tasks using LLMS.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gartu/LLMTaskKit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "litellm",
        "pydantic",
        "pytest",
        "python",
        "PyYAML",
    ],
)
