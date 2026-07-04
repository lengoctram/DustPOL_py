import os
from setuptools import setup, find_packages

setup(
    name="DustPOL_py",  # The name of your package
    version="0.1.8",  # Initial version of the package
    author="Le N. Tram",
    author_email="lengoctramlyk31@gmail.com",
    description="modeling dust polarization",
    long_description=os.path.realpath('__file__'),#open("README.md").read(),  # Use README.md as long description
    long_description_content_type="text/markdown",
    url="https://github.com/lengoctram/DustPOL_py",
    packages=find_packages(),  # Automatically find all packages and sub-packages
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">3.9",
    install_requires=[  # List your project's dependencies here
        "numpy>1.18.5",
        "matplotlib>3.10.0",
        "astropy",
        "scipy",
        "joblib",
        "pwlf",
        "lmfit",
        "sympy",
        "pandas"
        # Add other dependencies as needed
    ],
    include_package_data=True,  # To include non-code files specified in MANIFEST.in
    # entry_points={  # Optional: specify console scripts if your package has CLI commands
    #     "console_scripts": [
    #         "my_project=my_project.main:main_function",  # command=package.module:function
    #     ],
    # },
)
