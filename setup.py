from setuptools import setup, find_packages

def parse_requirements(filename: str):
    with open(filename, "r") as f:
        lines = f.readlines()
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="wireworld-nca",
    version="0.7.1",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=parse_requirements("requirements.txt"),
)