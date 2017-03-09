# Neural Networks and Deep Learning (.com)

This repository is an example of working through the book at: http://neuralnetworksanddeeplearning.com

But the original book was in Python 2 and is showing its age a bit. I was totally unable to run it on my laptop, in Python 2.7 or 3.5

So I am re-creating the book project in a standard Python 3.5 build with Anaconda on Linux:

    Python 3.5.2 |Anaconda 4.2.0 (64-bit)| (default, Jul  2 2016, 17:53:06) 
    [GCC 4.4.7 20120313 (Red Hat 4.4.7-1)] on linux

Where possible, I will use Tensorflow do to things like: import the MNIST data set and as a replacement for Theano.


## Chapter 1 - Neural Network Introduction with MNIST

Loading the MNIST data was a complete distaster. There were 32-bit vs 64-bit problems, problems with the version of Python and various `pickle` vs `cPickle` versions.

The new version of the chapter 1 scripts pull the MNIST data directly from TensorFlow. This is a lot more reliable (Google backing) and I am including the data just to be sure there are no issues.

I also converted the project to Cython, to improve performance. I only did a cursory compile-change though. I did not go through the code line-by-line and re-write it.
