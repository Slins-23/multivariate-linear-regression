# Multivariate Linear Regression


This is a Python implementation from scratch of multivariate (or univariate) linear regression, through gradient descent. It is my first machine learning project as I spent a few days studying linear regression as an introduction to the field. Expect inaccuracies and lack of optimization.

<p align="center">Example showcase video (bad quality because Github limit is 10MB)</p>


https://github.com/Slins-23/multivariate-linear-regression/assets/35003248/f80c510b-5911-4f9d-955f-49240749198a

## Notes
**This script only interprets datasets in `.csv` format (not necessarily extension), with comma-separated columns and newline separated rows. It ALWAYS interprets the columns as each being a potential feature, and the rows as each being a potential entry. If your dataset has each row as a feature and each column as an entry, it will not work properly!**

As I made this in a few days time with limited knowledge on the subject, I tried to balance speed, readability, ease of use, and customization as much as possible, so there are sections which are deliberately redundant, slower, and/or not customizable. Also keep in mind that the structure and features of the script were thought of as I kept coding (in realtime), nothing (other than getting univariate linear regression to work with this specific dataset) was planned. 

## Features
* Univariate or multivariate
* Normalization
* Euclidean distance between reference values and the dataset (numbers only)
* Filtering dataset

## Implementation details
The normalization occurs to the range [0, 1].

Assume that 
<br>
`K = number of independent features given by the user when prompted`
<br>
`N = number of entries`

There are 3 main matrices in which the algorithm plays out. They are the variables `B`, `X`, and `P`. 

<p align="center"> Weight Matrix (B) </p>
    <p align="center">Each row represents a weight</p>
        <p align="center">(Numpy indexes at 0)</p>
        <p align="center">k = Number of (independent) features</p>
              <p align="center">(k + 1) x 1</p>
        <p align="center">[ b0 ]</p>
        <p align="center">[ b1 ]</p>
        <p align="center">[ b2 ]</p>
        <p align="center">[ .  ]</p>
        <p align="center">[ .  ]</p>
        <p align="center">[ .  ]</p>
        <p align="center">[ bk ]</p>

`B` represents the weight matrix (coefficients of the predicted model). Its dimensions are `(K + 1)x1`. It can be predefined or randomly initialized, the option is given in a prompt.

<p align="center">Input Matrix (X)</p>
    <p align="center">Each entry is a column vector</p>
        <p align="center">(Numpy indexes at 0)</p>
        <p align="center">k = Number of (independent) features</p>
        <p align="center">n = Number of entries</p>
              <p align="center">(k + 1) x n</p>
        <p align="center">[ 1    1  . . . .  . .  . . .   1 ]</p>
        <p align="center">[ x11 x22 . . . x2n ]</p>
        <p align="center">[ x21 x32 . . . x3n ]</p>
        <p align="center">[ . . . . . . . . .   .  . . .   . ]</p>
        <p align="center">[ .    .  . . .  . . . . . . . .  . ]</p>
        <p align="center">[ .    .  . . . . . . . . . . .   . ]</p>
        <p align="center">[ xk1 xk2 . . . xkn ]</p>

`X` represents the entries matrix. Its dimensions are `(K + 1)xN`, where `K` is the number of independent features given by the user in the prompt, and `N` is the number of entries. Each entry is a column vector, and each subsequent row is a feature of that vector, except the first row, as it is comprised of only 1s. This is because `B`, the weight matrix, eventually gets multiplied with the weight matrix `X`, and the first element/weight in `B` is a constant.

<p align="center">Gradient vector (P)</p>
    <p align="center">Each row represents a partial derivative</p>
<p align="center">(with respect to the respecitve weight variable)</p>
        <p align="center">(Numpy indexes at 0)</p>
    <p align="center">k = Number of (independent) features</p>
    <p align="center">n = Number of entries</p>
          <p align="center">(k + 1) x 1</p>
            <p align="center">[ p0 ]</p>
            <p align="center">[ p1 ]</p>
            <p align="center">[ p2 ]</p>
            <p align="center">[ .  ]</p>
            <p align="center">[ .  ]</p>
            <p align="center">[ .  ]</p>
            <p align="center">[ pk ]</p>

`P` represents the gradient vector. Its dimensions are `(K + 1)x1`. Each row contains the partial derivative of the mean squared error with respect to the respective component in the predicted line equation.

Each matrix is initialized as an `ndarray` and subsequently reassigned to a matrix with itself as the argument.

The learning rate, `LR`, is given by the user, and is eventually used to multiply the gradient vector.

The variable `TOTAL_STEPS` gets assigned to the number of training steps given by the user in the prompt.

For every step, the `minimize` function gets executed. There, the expected value matrix is retrieved as the result of the operation: `B.transpose() * X`. This result is then transposed once again, as it had dimensions `1xN` (row vector), now it has `Nx1` (column vector). Each row in this matrix represents the expected value of the entry at the respective column in matrix `X`.

Then we get the error matrix, which is the result of subtracting the expected value matrix from the correct value matrix: `ERROR_MATRIX = EXPECTED_VALUE_MATRIX - TARGET_VALUES_MATRIX`.

The gradient vector then gets set to `X` multiplied by the error matrix: `P = X * ERROR_MATRIX`. The result then gets divided by the `scalar` value `N` (number of entries), in order to get the mean: `P /= N`.

Then update the weight matrix `B` to the matrix `B` minus the learning rate `LR` times the gradient vector `P`: (`B -= LR * P`). We subtract because the gradient vector points to the direction of steepest ascent, but we want to descend.

The mean squared error is then calculated as the dot product of the error matrix: `mse = ERROR_MATRIX.transpose() * ERROR_MATRIX`, which is then divided by the scalar `2 * N`: `mse / (2 * N)`, which gets subsequently appended to a list, in order to be graphed later. The error is divided by `2` because the derivative of the error function will always be multiplied by `2` as the error function is squared by definition, and that power becomes a constant multiplier per the chain rule when taking the derivative of the error function. The error function I use already divides this two, so that the gradient vector does not have any 2s. This means that the error function is actually half of the mean squared error, but the exact error doesn't really matter as what is important here is the behavior over time (i.e. when it becomes stable and stops decreasing).

Then, once training has started, a graph of the loss function will be plotted, and updated, in as many milliseconds as have been provided in the prompt that asks for the interval. These two things occur in parallel, in separate threads.

This is the case for all types of linear regression models, but if the model happens to be univariate, there will also be a 2D plot alongside the graph of the loss function, which will contain a scatter plot of all the entries and the predicted line as the model is training, which is updated at the given interval. If you have also provided a reference value, the points farthest from your reference will be color coded as `red`, and the closest will be color coded as `green`. This is calculated through euclidean distance.

As the distance comparison between reference values and the dataset is done using euclidean distance, I have restricted the script to work only with numeric values.

The reference values are compared against the dataset before it gets normalized (if you chose to normalize).

## Todo
### Issues
* There is some fluctuation in the matrix of partial derivatives (gradient vector) as the weights become more stable. I need to find a more consistent way of displaying them and/or fix this fluctuation;

* Fix the race conditions and subsequent bugs caused by training the model and updating the plot in parallel;

* If reference values are given for the distance calculation and you choose to filter the dataset, there will be an option added by the script called "distance_builtin" as if it were a column or feature of the dataset. It represents the calculated euclidean distance. If you have a column in your given dataset with this same name and reference values were given, the script might not work correctly;

* There are currently many prompt settings, and this is a design choice as I wanted to make the script relatively general and give the user a certain amount of control. In direct contrast, one consequence of this choice is the ease of use and clarity of instructions. The amount of settings I exposed to the user are inversely proportional to its ease of use. Which becomes even more apparent in the command line. This makes it relatively error-prone, especially as you get deeper into the configuration steps, and you often have to quit the script if you get a single setting wrong since I haven't implemented ways to "undo" a prompt. A web interface or GUI would make it much easier to use, but this was not the focus of this project.

### Features

* Implement cosine similarity for strings when comparing the distance between reference value(s) and the dataset;

* Add yet another user prompt to control when to compare the reference values to the dataset (before or after normalization);

* Add option for mean normalization;

* Allow user to predict values without having to close the plot.

## Motivation

I wasn't planning on uploading this, but as I was making the script with the primary goal of getting univariate linear regression to work with a specific dataset, I ended up implementing quite a few features that generalized it more, so that anyone can use it. I also tend to upload my first project(s) when studying a new topic, so I figured this would be fitting.





