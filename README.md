# Allison
Allison: is a library  of Artificial Intelligence

# About Allison

This project implements the main machine learning and deep learning architectures in Python, 
using libraries such as Numpy, Scipy and Matplotlib for the visualizations. 
its objective is to show how machine learning works
inside and not be a black box for many people.

## Requirements
- Numpy
- Matplotlib
- Scipy
- Pandas
## Install
- clone the repository `git clone https://github.com/Mitchell-Mirano/Allison.git`
- install a virtual environment `python -m virtualenv env`
- install the requirements `pip install -r requirements.txt`

Now you can use Allison

## Example of Allison in Google Colab
- Clone the repository
- import the requirements
- import  Kmeans model  and select your features for train.

![Screenshot_100](https://user-images.githubusercontent.com/67351445/116353342-33d1a680-a7bc-11eb-8a34-5e0381d9ba3b.png)

- create a model object with the Kmeans class and use the train method for trainning.
- the moviment_limit param is the condition for stop the training when the mean of the movements of the centroids is < movement limit
- you can use the centroids and labels atribute for obtain the new centroids and labels
- you can save the labels with save_labels method

![Screenshot_82](https://user-images.githubusercontent.com/67351445/115980467-94f64180-a552-11eb-8053-43b18ea40823.png)


- Graph your features with your new labels and centroids
![Screenshot_83](https://user-images.githubusercontent.com/67351445/115980461-8740bc00-a552-11eb-9750-f4492db70e7f.png)


