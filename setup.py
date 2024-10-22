from setuptools import find_packages, setup

# parse requirements.txt and ignore comments
with open("requirements.txt") as f:
    required = f.read().splitlines()
    required = [r for r in required if not r.startswith("#") and not r.startswith("-r")]


setup(
    name="cyber",
    version="0.1",
    packages=find_packages(include=["cyber"]),
    install_requires=required,
)
