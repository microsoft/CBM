import sysconfig

from setuptools import setup
from setuptools.extension import Extension


class get_pybind_include(object):
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11

        return pybind11.get_include(self.user)

def get_extra_compile_args():
    cflags = sysconfig.get_config_var("CFLAGS")
    if cflags is None:
        cflags = ""

    return cflags.split() \
            + ["-std=c++11", "-Wall", "-Wextra", "-march=native", "-msse2", "-ffast-math", "-mfpmath=sse"]

setup(
    name="cyclicbm",
    version="0.0.2",
    description="Cyclic Boosting Machines",
    url="https://github.com/Microsoft/CBM",
    author="Markus Cozowicz",
    author_email="marcozo@microsoft.com",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    setup_requires=["pytest-runner"],
    install_requires=["pybind11>=2.2", "numpy", "scikit-learn"],
    tests_require=["pytest", "lightgbm", "interpret"],
    packages=["cbm"],
    ext_modules=[
        Extension(
            "cbm_cpp",
            ["src/pycbm.cpp", "src/cbm.cpp"],
            include_dirs=[get_pybind_include(), get_pybind_include(user=True)],
            extra_compile_args=get_extra_compile_args(),
            libraries=["stdc++"],
            language="c++11",
        )
    ],
    zip_safe=False,
)