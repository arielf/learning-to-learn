# Learning to learn with TensorFlow playground

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


## A trivial classifier

Let's start with the simplest problem: 2 diagonally opposite clusters.

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
  - The boundary between the orange and blue clusters is smooth & clear:
    - It lies half-way between the closest points in the two sets (the support vectors)
    - Its orientation (from corner to corner) is nearly perfect and perpendicular to the line between the closest-points _and_ the cluster centroids.

## 4-clusters with x1, x2 overlap

Continuing with the 2nd simplest problem:

![Simple Classifier Solution](images/002-simple-classifier.png)

Again, the two clusters are clearly separable in the original input space.
but the boundary is more complex since we now have 4 instead of 2 clusters.

We need two lines, one horizontal and one vertical to achieve the separation.

The choice of the input feature is critical: leaving the original
input features would result in a complete inability to learn. The
learning curves would be moving sideways rather than dropping and
converging towards zero.

The best input feature is the multiplicative feature (`x1 * x2`)
instead of the original coordinates.

***Notes:***

  - Again once we pick the best and most relevant input feature, we get to near zero losses on both train and test sets after less than 500 iterations
  - Again, there's no need to use hidden layers
  - Again convergence to a great solution is both fast and smooth
  - The test and training loss end-up equal
  - Because the boundaries between the clusters are much narrower, choices of smaller values for both the learning & regularization rates are helpful
  - The input feature choice is the most critical choice for a great solution

### Final test and test error
### Convergence speed
### Smoothness of the learning curve
### Train vs test gap
### Clean boundaries between classes
### Overfitting vs underfitting
### Other generalization considerations

## Importance of the input features

### The concentric spiral problem
#### coordinate projections can make a big difference

WIP ...
