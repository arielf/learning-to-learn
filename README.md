# Learning to learn with TensorFlow playground (Spoilers ahead!)

## Introduction to tensor-flow playground
[The tensor-flow playground](https://playground.tensorflow.org/) is a wonderful tool to experiment with simple neural nets.

Given a small number of problems in 2-dimensions (x1, x2),
you're asked to design a good classifier / regressor model that will
correctly predict the (output) value of every point in input space
by tinkering with various model parameters.

Making the right model-design decisions can result in dramatic differences
in the model quality.

The challenge is to create the best model possible for each of the problems
presented.

***Note:*** these are spoilers. A set of "cheats" for near perfect
solutions. You're encouraged to try and solve the problems yourself,
before looking at the solutions below.

## Problem #1: a trivial classifier in 2D

Let's start with the simplest problem: 2 diagonally opposite clusters
in two dimensions `(x1, x2)`.

The challenge is to design a learner that will separate the
orange vs. blue clusters in an optimal way.

With the right input-features & parameter choices we get a perfect
classifier very quickly.

![Trivial Classifier Solution](images/001-simplest-classifier.png)

The two clusters are clearly and directly linearly-separable
in the original input space.

***Notes:***

  - There's no need to transform any original feature
  - Less is more: two input features (`x1`, `x2`) are sufficient
  - No need to add any hidden layer: the input and output layers can be directly connected
  - We can learn fast, using a high learning rate without fear of early overfitting

## How can we tell we have a good/great model?

  - Both learning curves (train & test) drop sharply and quickly towards near zero
  - We see no upward-spikes in the learning curves
  - Final test loss is lower than train loss; this is very unusual/rare. It indicates that our model is not too complex nor too simple and has great generalization power.
  - The **boundary** between the orange and blue clusters is smooth & clear:
    - It lies half-way between the closest points in the two sets (the support vectors)
    - Its orientation (from corner to corner) is nearly perfect and perpendicular to the line between the closest-points _and_ the cluster centroids.

## Problem #2: 2-pairs of clusters with x1, x2 overlap

Continuing with the 2nd simplest problem, 4-clusters with `(x1, x2)`
overlap.

Again, the clusters are clearly separable in the original input space,
But the boundary is more complex since we now have 4 instead of 2
clusters, where each pair of same-color clusters lies along one of the
diagonals.

We need two lines, one horizontal and one vertical to achieve the separation.

The choice of the input feature is critical: keeping the original
input features (`x1, x2`) results in a complete inability to learn.
The learning curves are moving sideways rather than dropping and
converging towards zero as can be seen here:

![Inability to converge](images/003-unable-to-converge.png)

We could "add capacity" complicating the model by adding
hidden-layers. This allows the training to converge to a (near zero loss) solution, but this solution is not optimal:

![Adding capacity/layers to the network](images/004-adding-capacity.png)

***How can we tell the solution isn't optimal?***

  - The test loss is consistently ***much*** higher than the training loss
  - The class-boundaries are not as we expect them (along the 4 quadrants): note the rhombus (instead of square) shape of the upper-right quadrant.
  - Over-fitting is in effect: whenever we click on the "regenerate" button (creating slightly different data-sets) the boundaries of the model shift. This tells us that the [generalization error](https://en.wikipedia.org/wiki/Generalization_error) is poor. IOW: the expected out-of-sample error is large.
  - The hidden-layers (partial solutions):
    - Have a lot of overlaps/redundancies between them
    - Are not optimally oriented (vertically, horizontally, diagonally)

Realizing that the best and most-relevant input feature is the
multiplicative feature instead of the original coordinates,
we switch from the two original coordinates (`x1, x2`)
to a single multiplicative input feature (`x1 * x2`).
This projection of the input space to a new coordinate system
perfectly describes the 4-quadrant mapping into the two diagonal
sets, and results in a near perfect solution without any need to
increase model-complexity (add capacity in the form of hidden layers):

![Simple Classifier Solution](images/002-simple-classifier.png)

***Notes:***

  - Again once we pick the best and most relevant input feature, we get to near zero losses on both train and test sets after less than 500 iterations
  - Again, there's no need for hidden layers
  - Again, convergence to a great solution is both fast and smooth
  - The test and training loss are very close during training and end-up equal to each other
  - Because the boundaries between the clusters are much narrower, choices of smaller values for both the learning & regularization rates are helpful
  - The input feature choice is the most critical choice for a great solution

## Problem #3: Concentric Circles

## Problem #4: Concentric (and overlapping) Spirals

## Indicators of good/great/perfect solutions

### Final test and test error
### Convergence speed
### Smoothness of the learning curve
### Train vs test gap
### Clean boundaries between classes
### Over-fitting vs under-fitting
### Other generalization considerations

## Importance of the input features

### Coordinate projections

## !!! Work in Progress !!!

