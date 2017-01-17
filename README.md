# Neural Networks and Deep Learning (.com)

I am simply working through the entire book at http://neuralnetworksanddeeplearning.com

But the original book was in Python 2 and is showing its age a bit. I was totally unable to run it on my laptop, in Python 2.7 or 3.5, so I am re-creating the book project in a standard Python 3.5 build with Anaconda on Linux:

    Python 3.5.2 |Anaconda 4.2.0 (64-bit)| (default, Jul  2 2016, 17:53:06) 
    [GCC 4.4.7 20120313 (Red Hat 4.4.7-1)] on linux


## Chapter 1 - Neural Network Introduction with MNIST

Loading the MNIST data was a complete distaster. There were 32-bit vs 64-bit problems, problems with the version of Python and various `pickle` vs `cPickle` versions.

The new version of the chapter 1 scripts pull the MNIST data directly from TensorFlow. This is a lot more reliable (Google backing) and I am including the data just to be sure there are no issues.
