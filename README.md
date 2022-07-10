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

Let's start with the simplest classifier. The challenge is to
design a learner that will separate the orange vs. blue clusters
in an optimal way. With the right input-features and parameter choices
we get a perfect classifier very quickly.

![images/001-simplest-classifier.png]

Since the two sets of points are clearly and directly
linearly-separable in the original space. Note that:

  - There's no need to transform any original feature
  - Less is more: two input features (`x1`, `x2`) are sufficient
  - No need to add any hidden layer: the input and output layers can be directly connected
  - We can learn fast, using a high learning rate without fear of early overfitting
  - Test loss is lower than train loss; this is rare and indicates that
    our model is not too complex nor too simple and has great generalization power.
  - The boundary between the clusters is smooth, clear:
   - It lies half-way between the sets
   - Its orientation (from corner to corner) is nearly perfect.

## How can we tell we have a good/great model?


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
