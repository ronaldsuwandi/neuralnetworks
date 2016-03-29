(ns neuralnetworks.optimizer)

(defprotocol Optimizer
  "Protocol for the optimizer that can be used to minimize the given cost function.

   Various optimizer can have its own options (stopping conditions, additional parameters)"
  (optimize [this cost-fn thetas]
    "Optimize the given cost function for thetas parameters. Thetas is a vector and cost function
     will be responsible to transform it into proper matrices

     Will return a map of cost value and the new theta
     {:cost 1.56
      :thetas [...]"))
