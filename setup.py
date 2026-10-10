from pathlib import Path
from setuptools import find_packages, setup

setup(
    name="marlim",
    version="1.0.0",
    description="Multi-agent inventory and transshipment experiments",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=Path(__file__).with_name("requirements.txt").read_text().splitlines(),
    include_package_data=True,
)
