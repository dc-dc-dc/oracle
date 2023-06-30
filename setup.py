import os 
from setuptools import setup

setup(
    name="oracle",
    version="0.0.1",
    author="Diogo Da Cruz",
    author_email="diogodacruz.dev@gmail.com",
    packages=['oracle', 'oracle.export', 'oracle.system'],
    description=("A machine learning analytics utility to track your training performance"),
    license="MIT",
    keywords="machine learning analytics pytorch tinygrad",
    long_description=open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding="utf-8").read(),
    long_description_content_type='text/markdown',
    install_requires=["numpy"],
    include_package_data=True,
    python_requires='>=3.8',
)