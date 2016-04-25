# Change Log

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
