# neuralnetworks

[![Clojars Project](https://clojars.org/ronaldsuwandi/neuralnetworks/latest-version.svg)](https://clojars.org/ronaldsuwandi/neuralnetworks)

[![Build Status](https://travis-ci.org/ronaldsuwandi/neuralnetworks.svg?branch=master)](https://travis-ci.org/ronaldsuwandi/neuralnetworks) [![Dependency Status](https://www.versioneye.com/user/projects/57066022fcd19a004543fcfd/badge.svg?style=flat)](https://www.versioneye.com/user/projects/57066022fcd19a004543fcfd)

Neural networks library for Clojure. Built on top of [core.matrix](https://github.com/mikera/core.matrix) 
array programming API 

This library is created as proof of concept that neural networks can be done using Clojure.

Currently it has the following features

- Regularization
- Swappable optimizer. Currently it only supports [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent) 
  with [Backtracking Line Search](https://www.cs.cmu.edu/~ggordon/10725-F12/slides/05-gd-revisited.pdf).
  More optimizer will be added in the future (e.g. Evolutionary Algorithms)
- Multiple stopping conditions. Currently it supports stopping conditions based on error or number
  of iterations. If multiple stopping conditions are provided, it will be treated as `OR` (if either
  stopping condition is fulfilled, the optimizer stops training)
- Swappable activation/sigmoid function. Currently it has 2 functions:
  - [Standard logistic function](https://en.wikipedia.org/wiki/Logistic_function)
  - Optimized hyperbolic tangent function. Reference: [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
  
**Note - this library is not yet production ready**

## Usage

The following is an example of how to use neural networks library to train for `AND` function

```clojure
(require '[neuralnetworks.core :as nn])
(require '[clojure.core.matrix :as m])

(let [input    (m/array [[0 0]
                         [0 1]
                         [1 0]
                         [1 1]])
      thetas   (nn/randomize-thetas 2 [3] 1)
      output   (m/array [[0]
                         [0]
                         [0]
                         [1]])
      options  {}
      instance (nn/new-instance input thetas output options)]

  (prn "Before training: " (nn/predict instance input))
  (nn/train! instance)
  (prn "After training: " (nn/predict instance input)))
```

If an empty map is provided as the options, then the default settings are used. Currently these are 
the available options

* `:learning-rate` (or alpha) - default value is 1.0
* `:regularization-rate` (or lambda) - default value is 0.0
* `:sigmoid-fn` - default value is standard logistic function
* `:optimizer` - default value is gradient descent with the following settings
    * learning rate of 8
    * learning rate update rate of 0.5
    * single stopping conditions of 100 iterations

Example of options

```clojure
(use '[neuralnetworks.sigmoid-fn])
(use '[neuralnetworks.optimizer.gradient-descent])
(use '[neuralnetworks.stopping-conditions])

(def options {:sigmoid-fn standard-logistic
              :stopping-conditions [(max-iterations 100)]
              :optimizer (gradient-descent 8 0.5 [(max-iterations 100)]})
```
