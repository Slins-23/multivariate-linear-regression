# Multivariate Linear Regression


This is a Python implementation from scratch of multivariate (or univariate) linear regression, through gradient descent. It is my first machine learning project as I spent a few days studying linear regression as an introduction to the field. Expect inaccuracies and lack of optimization.

<p align="center">Example showcase video (bad quality because Github limit is 10MB)</p>


https://github.com/Slins-23/multivariate-linear-regression/assets/35003248/f80c510b-5911-4f9d-955f-49240749198a

## Notes
As I made this in a few days time with limited knowledge on the subject, I tried to balance speed, readability, ease of use, and customization as much as possible while I wrote this, so there are sections which are deliberately redundant, slower, and/or not customizable. Also keep in mind that the structure and features of the script were thought of as I kept coding in realtime, nothing (other than getting univariate linear regression to work with this specific dataset) was planned. 

## Features
* Univariate or multivariate
* Accepts datasets in .csv format with comma-separated columns
* Normalization
* Euclidean distance between reference values and the dataset (numbers only)
* Filtering dataset

## Implementation details
The distance comparison between reference values and the dataset is done using euclidean distance, and as such I have restricted the script to work only if numeric values and not strings when it comes to them.

The reference values are compared against the dataset before it gets normalized (if you chose to normalize).

## Todo
### Issues
 * There is some fluctuation in the matrix of partial derivatives (gradient vector) as the weights become more stable, I need to find a more consistent way of displaying them and/or fix this fluctuation;

* There are currently many prompt settings, and this is a design choice as I wanted to make the script relatively general and give the user a certain amount of control. In direct contrast, one consequence of this choice is the ease of use and clarity of instructions. The amount of settings I exposed to the user are inversely proportional to its ease of use. Which can be even more apparent in the command line. This makes it relatively error-prone, especially as you get deeper into the configuration steps, and you often have to quit the script if you get the settings wrong, I haven't implemented ways to "undo" a prompt. A web interface or GUI for this make it much easier to use, but this was not the focus of this project.

### Features

* Implement cosine similarity for strings when comparing the distance between reference value(s) and the dataset;

 * Add yet another user prompt to control when to compare the reference values to the dataset (before or after normalization);

## Motivation

I wasn't planning on uploading this, but as I was making the script with the primary goal of getting univariate linear regression to work with a specific dataset, I ended up implementing quite a few features that generalized it more, so that anyone can use it. I also tend to upload my first project(s) when studying a new topic, so I figured this would be fitting.





