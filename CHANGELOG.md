# Change Log

## 0.3.0-SNAPSHOT [Unreleased]
### Added
- Added error function to measure the performance between expected output and predicted output
- Added new logging dependencies. Setting the level to `:trace` will print the error value for each
  iteration

## 0.2.1
### Changed
- Removed codox factory function documentation for defrecord

## 0.2.0
### Added
- New activation/sigmoid function (hyperbolic tangent)
- New cost function (mean squared error)
- Cost function accepts varargs and currently responds to `:skip-gradient`. This allows line search
  to be performed without having to calculate gradients for each `alpha` to improve performance

### Changed
- Renamed activation function to sigmoid function
- Cost/error function is now swappable
- New instance accepts problem type (classification or regression) and chooses the default cost and
  sigmoid functions accordingly

### Removed
- Binary sigmoid function

## 0.1.1-SNAPSHOT
### Added
- Line search for gradient descent optimizer

## 0.1.0-SNAPSHOT
Initial release
