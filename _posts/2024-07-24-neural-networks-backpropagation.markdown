---
layout: post
title:  "Neural Networks Backpropagation"
date:   2024-07-24 11:08:03 +0200
categories: ai
---

# Gradient

[Video](https://www.youtube.com/watch?v=YS_EztqZCD8)

The gradient captures all the partial derivative information of a scalar-valued multivariable function. Created by Grant Sanderson.

A vector of partial derivatives for a multivariate function.

Gives the direction of steepest ascent (descent) of the function.

## The directional derivative

The directional derivative is the dot product between the gradient and the unit vector in that direction.

In our case we have the C as the cost function, and the partial derivatives for the weights and biases. We want to descent the gradient of the cost function.

The dimensionality of the gradient space is given by the number of weights and biases for the model.

## The chain rule

The chain rule tells us that the derivative of a composite function is equal to the product of the derivatives of each of its parts.

df(g(x))/dx = df/dg * dg/dx

We have a cost function L, and we want to find the partial derivative of L with respect to each parameter. We can do that by using the chain rule:

if L = f*g, and f=h+k => dL/df = g, dL/dg = f, df/dh = 1, df/dk = 1. Using the chain rule, dL/dh = dL/df*df/dh and so on. dL/dh is the gradient of h; how much does h impact the gradient descent.

Remarks:
* a plus sign distributes the gradient of a parent to its children.
* we can only influence leaf nodes during gradinet descent. In the example above, we can only influence h,k and g
* because a parameter can be referenced more than once, the gradients have to be summed up instead of overwritted at parameter level.

# Neuron

We have n inputs, x-es each with a weight, w-s. And a bias b. Then we have an activation function f, a squashing function. The value of the neuron is f(sum(xi*wi) + b).

# Layer

A set of n neurons

# MLP: multi-layer perceptron

A chaining of multiple layers: An input layer, 0 to multiple hidden layers and the output layer. Each neuron in Layer n is connected to each neuron in Layer n-1.

A forward pass: we take a set of input values and forward pass through the entire network. There's an activation function at the end with the main goal of squashing the values. Why do we need squashing: to make sure that the output is bounded between 0 and 1. We call the output of this layer the activations. Multiple samples are processed in parallel in a batch and a loss or cost function is computed over the predictions of each sample versus the extected values.

Backward propagation is called on the loss function to calculate the gradients for each parameter over the entire batch. Based on the gradients, we update the parameters in the direction that reduces the loss (the gradient descent).

# How to choose a proper learning rate?

Instead of a static learning rate, build a dynamic learning rate with the powers of 10 between -3 and 0; 1000 of them

```
lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre
```

This will be between 0.001 and 1, but exponentiated.
![alt text](image.png)

Run a training loop with the dynamic learning rate, save the loss and plot it. You get something like this:
![alt text](image-1.png)
So the best rate is between the -1 and -0.5 exponent of 10.