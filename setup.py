import setuptools

install_deps = [
        "numpy>=1.24.0", 
        "scipy", 
        "scikit-learn", 
        "numba>=0.57.0",
        "natsort",
        "tqdm"
        ]

gui_deps = [
        "pyqtgraph>=0.11.0rc0", 
        "pyqt5", 
        "pyqt5.sip",
        "superqt",
        ]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rastermap",
    version="0.1.3",
    author="Marius Pachitariu and Carsen Stringer",
    author_email="carsen.stringer@gmail.com",
    description="Unsupervised clustering algorithm for 2D data (neurons by time)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MouseLand/rastermap",
    packages=setuptools.find_packages(),
	install_requires = install_deps,
    extras_require = {
      "gui": gui_deps
    },
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ),
)
