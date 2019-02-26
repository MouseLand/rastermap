import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rastermap",
    version="0.0.6",
    author="Marius Pachitariu and Carsen Stringer",
    author_email="carsen.stringer@gmail.com",
    description="Unsupervised clustering algorithm for 2D data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MouseLand/rastermap",
    packages=setuptools.find_packages(),
	install_requires = ['numpy>=1.13.0', 'scipy', 'pyqtgraph', 'PyQt5', 'PyQt5.sip','matplotlib'],
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ),
)
