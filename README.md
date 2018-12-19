# OpenCV-Python Lane Detection

Simple python script to test common image processing techniques to detect road lane.  
Check Wiki section for an in-depth explanation of the techniques used.

### Prerequisites

The script is written in Python 3.7 but any version since 3.0 should work without problem.

You can download Python from the official website: [Python](https://www.python.org)  
For a better testing of the script an IDE may also come in handy, [Pycharm](https://www.jetbrains.com/pycharm) community edition was used. 

### Installing

* Install Python 3.x from official site and during the install wizard select to install ```pip```, a Python package installer.  
On windows may also be useful to add both Python and ```pip``` to PATH variable, to easily call them from command line.  
After you have Python additional modules are needed:
* [Numpy](http://www.numpy.org/) A powerful module for scientific computing with Python.  
* [OpenCV-Python](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_setup/py_intro/py_intro.html#intro) OpenCV-Python is a Python   wrapper for the original OpenCV C++ implementation.

To install those modules you can simply run the following into a command-line/terminal:
```
pip install numpy
pip install opencv-python
```
You can run ```pip list``` to check if the packages actually got installed.

## Running the tests

At this point you can simply point the command-line/terminal to the python script location and run:

Windows: ```python MainScript.py```  
Linux: ```python3 MainScript.py```
