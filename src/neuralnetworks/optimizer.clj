(ns neuralnetworks.optimizer)

(defprotocol Optimizer
  "Protocol for the optimizer that can be used to minimize the given cost function.

   Various optimizer can have its own options (e.g. additional parameters)"
  (optimize [this cost-fn thetas stopping-conditions]
    "Optimize the given cost function for thetas parameters. Thetas is a vector and cost function
     will be responsible to transform it into proper matrices.

     Cost function will also accept varargs (e.g. a flag to disable calculating gradient, etc)

     Stopping conditions is a vector of stopping condition functions used by the optimizer which in
     turn used by neural networks training function.

     If multiple stopping conditions are provided, it will be treated as *OR* meaning as long as one
     of the condition is satisfied, training will be stopped (i.e. optimizer is finished)

     Will return a map of cost value and the new theta
     {:cost 1.56
      :thetas [...]"))
