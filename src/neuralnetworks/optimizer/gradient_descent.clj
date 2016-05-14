(ns neuralnetworks.optimizer.gradient-descent
  (:require [clojure.core.matrix :as m]
            [taoensso.timbre :as log]
            [neuralnetworks.utils :refer [approx]]
            [neuralnetworks.optimizer :refer :all]))

(defn update-thetas
  "Updates the thetas (weight) based on the provided gradients and alpha.
   Gradients must be the same dimension as thetas"
  [thetas theta-gradients alpha]
  (mapv #(m/sub %1 (m/mul %2 alpha)) thetas theta-gradients))

(defn line-search
  "Uses backtrack line search algorithm to find the best learning-rate (alpha) value.

   Backtrack will stop if either of the following conditions are met:

   * Cost of new theta (updated theta - i.e. theta is updated with alpha and gradients) is less or
     equals to cost of the original theta minus gradient length squared with alpha
   * Approximately alpha reaches almost zero (1e-10)
   * Approximately all gradients almost zero (1e-10)

   Reference: [Gradient Descent Revisited](https://www.cs.cmu.edu/~ggordon/10725-F12/slides/05-gd-revisited.pdf)"
  [cost-fn init-alpha beta thetas theta-gradients]
  (if (approx beta 1.0 1e-10)
    init-alpha
    (let [zero-matrix (m/zero-vector (m/length theta-gradients))
          cost-theta (:cost (cost-fn thetas :skip-gradient))]
      (loop [alpha init-alpha]
        (let [theta-with-gradient (m/sub thetas (m/mul alpha theta-gradients))
              cost-theta-with-alpha (:cost (cost-fn theta-with-gradient :skip-gradient))
              gradient-length-squard-with-alpha (-> theta-gradients
                                                    m/length-squared
                                                    (* alpha 0.5))]
          (if (or (<= cost-theta-with-alpha (- cost-theta gradient-length-squard-with-alpha))
                  (approx alpha 0.0 1e-10)
                  (m/equals theta-gradients zero-matrix 1e-10))
            alpha
            (recur (* beta alpha))))))))

(defrecord GradientDescent [initial-learning-rate learning-rate-update-rate]
  Optimizer
  (optimize [this cost-fn thetas stopping-conditions]
    (loop [iteration 0
           thetas thetas]
      (let [cost (cost-fn thetas)
            optimizer (assoc this :iteration iteration
                                  :error (:cost cost)
                                  :thetas thetas)]
        (when-not (contains? cost :gradients)
          (throw (ex-info "Gradients is not returned by the cost function " {:cost cost})))
        (if (some #(% optimizer) stopping-conditions)
          (select-keys optimizer [:iteration :error :thetas])
          (let [alpha (line-search cost-fn
                                   initial-learning-rate
                                   learning-rate-update-rate
                                   thetas
                                   (m/as-vector (:gradients cost)))]
            (log/tracef "Cost: %.6f (#%d)" (:cost cost) iteration)
            (recur (inc iteration)
                   (update-thetas thetas (m/as-vector (:gradients cost))
                                  alpha))))))))

(defn gradient-descent
  "Creates new instance of gradient descent optimizer. It uses Backtracking Line Search to find the
   good value of learning-rate (alpha) to allows it converge faster

   To disable backtracking line search, simply set the learning-rate-update-rate to 1.0

   Learning-rate-update-rate must be between (0, 1]

   Cost function must returns both `gradients` and `cost` value
   ```
   {:cost 1.5142
    :gradients [1.2 -0.5]}
   ```"
  [initial-learning-rate learning-rate-update-rate]
  {:pre [(> learning-rate-update-rate 0) (<= learning-rate-update-rate 1)]}
  (->GradientDescent initial-learning-rate learning-rate-update-rate))

(alter-meta! #'->GradientDescent assoc :no-doc true)
(alter-meta! #'map->GradientDescent assoc :no-doc true)
