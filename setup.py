import sysconfig

from setuptools import setup
from setuptools.extension import Extension
import platform

class get_pybind_include(object):
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11

        return pybind11.get_include(self.user)

def get_extra_compile_args():
    if platform.system() == "Windows":
        return ""

    cflags = sysconfig.get_config_var("CFLAGS")
    if cflags is None:
        cflags = ""

    return cflags.split() \
            + ["-std=c++11", "-Wall", "-Wextra", "-march=native", "-msse2", "-ffast-math", "-mfpmath=sse"]

def get_libraries():
    if platform.system() == "Windows":
        return []

    return ["stdc++"]

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="cyclicbm",
    version="0.0.7",
    description="Cyclic Boosting Machines",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/Microsoft/CBM",
    author="Markus Cozowicz",
    author_email="marcozo@microsoft.com",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    setup_requires=["pytest-runner"],
    install_requires=["pybind11>=2.2", "numpy", "scikit-learn"],
    tests_require=["pytest", "lightgbm"], #, "interpret"],
    extras_require={
        'interactive': ['matplotlib>=2.2.0'],
    },
    packages=["cbm"],
    # package_data={ "cbm": ["src/pycbm.h", "src/cbm.h"] },
    ext_modules=[
        Extension(
            "cbm_cpp",
            ["src/pycbm.cpp", "src/cbm.cpp" ],
            include_dirs=[get_pybind_include(), get_pybind_include(user=True), "src"],
            extra_compile_args=get_extra_compile_args(),
            libraries=get_libraries(),
            language="c++11",
        )
    ],
    headers=["src/pycbm.h", "src/cbm.h"],
    zip_safe=False,
)
