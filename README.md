# Probably Interesting Data

#### Authors
Zachary McGrath, Kevin Ray (github.com/kray10)

## Description
Using one of the models and distributions described in class, we were to model two 
of the data sets listed on https://www.kaggle.com/uciml/datasets. The two chosen
were datasets on breast cancer in wisconsin and the 'classic' data set on iris flowers.

## Running 

The environment used was Python 3.7

### Installation

Clone the repo
```
%> git clone https://github.com/zmcgrath96/probability_interesting_data
```
Install necessary libraries
```
%probably_interesting_data> pip3 install -r requirements.txt
```

This will install all the required libraries needed for running the project

### Running on data sets

There are two outward facing python files used to make it easier to call the modeling 
function without having to pass them in all in command line

Calling the modeling function from CLI instead of through the respective python files can be 
done as follows:

```
%probably_interesting_data> python main.py data/<dataset csv file> <categorical column> <data column 1> <data column 2>
```

Similarly, if one wishes to change the python file for the respective data set (i.e. either cancerDataset.py or irisDataset.py),
simply change the last three integer strings to that of the format shown above. Make sure to leave the first array entry an empty string

#### Wisconsin Breast Cancer Dataset

```
%probably_interesting_data> python cancerDataset.py
```
After running, a plot will present itself 
In order to end the program, simply close out of the scatterplot

#### Iris Flower Dataset

```
%probably_interesting_data> python irisDataset.py
```
After running, a plot will present itself 
In order to end the program, simply close out of the scatterplot

## Discussion
### How machine learning can be used to model the dataset
Machine learning can be used to model these datasets in order to try and find any sort of correlation.
Due to the fact that these datasets contain categorical columns, where different rows of data are tagged 
with belonging to some sort of category, a relationtionship between the variables (columns) and the categories
(rows) can be found with machine learning algorithms.

### Process
The process for this project was fairly straight forward:
	1. Create an outward facing function responsible for seperating the columns and only getting data wanted
	2. Write a function for our model (K-Means for our case)
	3. Ensure that the functions themselves we're not hardcoded for any type of dataset, but rather parameterized
	   and abstract enough to be used with any type of categorial, relational dataset

	Although we had an overall understanding of the algorithm at a high level, implementation details were the 
	primary roadblocks in this project. Due to this, we referenced https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
	for help when encountering such a situation

	The number of clusters to use was determined by the number of categories found in the categorical column

### Results

#### Wisconsin Breast Cancer Dataset
This dataset did not lend itself particularly well to clustering. When plotting and using the number of categories as 
the number of clusters, it seems almost as if an arbitrary line was drawn in the dataset. This could be because of the fact 
that all columns were inherently related. All fields were functions of radius, thus it appears as if there is just one large  
dataset instead of many smaller ones 

#### Iris Flower dataset

This dataset is a 'classic' dataset. The data associated with each category lend themselves well to very clear clusters.
Because of this, the results, when plotted on a scatterplot and colored, reveal exactly what one would expect: 3 clear clusters 
for just about every combination of two columns
