(ns neuralnetworks.optimizer.gradient-descent
  (:require [clojure.core.matrix :as m]
            [neuralnetworks.optimizer :refer :all]))

(defn update-thetas
  "Updates the thetas (weight) based on the provided gradients and alpha.
   Gradients must be the same dimension as thetas"
  [thetas theta-gradients alpha]
  (mapv #(m/sub %1 (m/mul %2 alpha)) thetas theta-gradients))

(defrecord GradientDescent [learning-rate stopping-conditions]
  Optimizer
  (optimize [this cost-fn thetas]
    (loop [iteration 0
           thetas thetas]
      (let [cost (cost-fn thetas)
            optimizer (assoc this :iteration iteration
                                  :error (:error cost)
                                  :thetas thetas)]
        (when-not (contains? cost :gradients)
          (throw (ex-info "Gradients is not returned by the cost function " {:cost cost})))
        (if (some #(% optimizer) stopping-conditions)
          (select-keys optimizer [:iteration :error :thetas])
          (recur (inc iteration)
                 (update-thetas thetas (m/as-vector (:gradients cost)) learning-rate)))))))

(defn gradient-descent
  "Creates new instance of gradient descent optimizer.

   Cost function must returns both `gradients` and `cost` value
   ```
   {:cost 1.5142
    :gradients [1.2 -0.5]}
   ```"
  [learning-rate stopping-conditions]
  (->GradientDescent learning-rate stopping-conditions))
