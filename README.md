# neuralnetworks

[![Build Status](https://travis-ci.org/ronaldsuwandi/neuralnetworks.svg?branch=master)](https://travis-ci.org/ronaldsuwandi/neuralnetworks)

[![Clojars Project](https://clojars.org/ronaldsuwandi/neuralnetworks/latest-version.svg)](https://clojars.org/ronaldsuwandi/neuralnetworks)

[![Dependency Status](https://www.versioneye.com/user/projects/57066022fcd19a004543fcfd/badge.svg?style=flat)](https://www.versioneye.com/user/projects/57066022fcd19a004543fcfd)

Neural networks library for Clojure. Built on top of [core.matrix](https://github.com/mikera/core.matrix) 
array programming API 

This library is created as proof of concept that neural networks can be done using Clojure.

Currently it has the following features

- Regularization
- Swappable optimizer. Currently it only supports [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent).
  More optimizer will be added in the future (e.g. Evolutionary Algorithms)
- Multiple stopping conditions. Currently it supports stopping conditions based on error or number
  of iterations. If multiple stopping conditions are provided, it will be treated as `OR` (if either
  stopping condition is fulfilled, the optimizer stops training)

## Note - this library is not yet production ready

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
* `:activation-fn` - default value is sigmoid function
* `:stopping-conditions` - default value is maximum iterations of 100
* `:optimizer` - default value is gradient descent

Example of options

```clojure
(use '[neuralnetworks.activation-fn])
(use '[neuralnetworks.optimizer.gradient-descent])
(use '[neuralnetworks.stopping-conditions])

(def options {:activation-fn sigmoid
              :stopping-conditions [(max-iterations 100)]
              :optimizer (gradient-descent (:learning-rate merged-options)
                                           (:stopping-conditions merged-options))})
```
