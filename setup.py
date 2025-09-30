from setuptools import setup, find_packages

setup(
    name='GD-TSE',
    version='0.0.1',
    packages=find_packages(where="src"),  # find all packages in src/
    package_dir={"": "src"},              # tell setuptools that packages are under src/
)