# Multivariate Linear Regression


This is a Python implementation from scratch of multivariate (or univariate) linear regression, through gradient descent. It is my first machine learning project as I spent a few days studying linear regression as an introduction to the field. Expect inaccuracies and lack of optimization.

<p align="center">Example showcase video (bad quality because Github limit is 10MB)</p>


https://github.com/Slins-23/multivariate-linear-regression/assets/35003248/f80c510b-5911-4f9d-955f-49240749198a

## Notes
*** This script only interprets datasets in `.csv` format (but not necessarily extension), with comma-separated columns and newline separated rows. It ALWAYS interprets the columns as each being a potential feature, and the rows as each being a potential entry. If your dataset has each row as a feature and each column as an entry, it will not work properly!

As I made this in a few days time with limited knowledge on the subject, I tried to balance speed, readability, ease of use, and customization as much as possible while I wrote this, so there are sections which are deliberately redundant, slower, and/or not customizable. Also keep in mind that the structure and features of the script were thought of as I kept coding in realtime, nothing (other than getting univariate linear regression to work with this specific dataset) was planned. 

## Features
* Univariate or multivariate
* Normalization
* Euclidean distance between reference values and the dataset (numbers only)
* Filtering dataset

## Implementation details
Assume that 
K = number of independent features given by the user when prompted
N = number of entries

There are 3 main matrices in which the algorithm plays out. They are the variables `B`, `X`, and `P`. 

`B` represents the weight matrix (coefficients of the predicted model). Its dimensions are `(K + 1)x1`. It can be predefined or randomly initialized, the option is given in a prompt.

`X` represents the entries matrix. Its dimensions are `(K + 1)xN`, where K is the number of independent features given by the user in the prompt, and N is the number of entries. Each entry is a column vector, and each subsequent row is a feature of that vector. Except the first row, the first row of this matrix is only comprised of 1s. This is because `B`, the weight matrix, eventually gets multiplied with the weight matrix `X`, and the first element of `B` is a constant.

`P` represents the gradient vector. Its dimensions are `(K + 1)x1`. Each row contains the partial derivative of the mean squared error with respect to the respective component in the predicted line equation.

Each matrix is initialized as an ndarray and subsequently reassigned to a matrix with itself as the argument.

The learning rate, `LR` is given by the user, and is eventually used to multiply the gradient vector.

The variable `TOTAL_STEPS` gets assigned to the number of training steps given by the user in the prompt.

For every step, the `minimize` function gets executed. There, the expected value matrix is retrieved as the result of the operation `B.transpose() * X`. This result is then transposed once again, as it had dimensions `1xN`, now it has `Nx1`. Each row in this matrix represents the expected value of the entry at the respective column in matrix `X`.

Then we get the error matrix, which simply subtracts the expected value matrix from the correct value matrix.

The gradient vector then gets set to `X` multiplied by the error matrix. The result then gets divided by the `scalar` value `N` (number of entries), in order to get the mean.

Then update the weight matrix `B` to the matrix `B` minus the learning rate `LR` times the gradient vector `P`. We subtract because the gradient vector points to the direction of steepest ascent, but we want to descend.

The mean squared error is then calculated and appended to a list, in order to be graphed later.

Then, once training has started, a graph of the loss function will be plotted, and updated, in as many milliseconds as have been provided in the prompt that asks for the interval. These two things occur in parallel in separate threads.

This is the case for all types of linear regression models, but if the model happens to be univariate, there will also be a 2D plot alongside the graph of the loss function, which will contain a scatter plot of all the entries and the predicted line as the model is training, which is updated at the given interval. If you have also provided a reference value, the points farthest from your reference will be color coded as `red`, and the closest to your reference will be color coded as `green`. This is calculated through euclidean distance.

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





