(ns neuralnetworks.core
  (:require [neuralnetworks.sigmoid-fn :refer [standard-logistic]]
            [neuralnetworks.calculate :as calculate]
            [neuralnetworks.optimizer :as optimizer]
            [neuralnetworks.optimizer.gradient-descent :refer [gradient-descent]]
            [neuralnetworks.optimizer.stopping-conditions :refer [max-iterations]]
            [clojure.core.matrix :as m]))

(defn train!
  "Train the neural networks. This will update the thetas/weights"
  [instance]
  (let [states (:states instance)
        thetas @(:thetas states)
        thetas-dimensions (mapv m/shape thetas)
        cost-fn (calculate/cost-fn (:input instance)
                                   (:output instance)
                                   (:sigmoid-fn instance)
                                   (:regularization-rate instance)
                                   thetas-dimensions)
        optimized (optimizer/optimize (:optimizer instance) cost-fn (m/as-vector thetas))]
    (reset! (:iteration states) (:iteration optimized))
    (reset! (:thetas states) (calculate/reshape-thetas (:thetas optimized) thetas-dimensions))
    instance))

(defn predict
  "Predict the output given the neural networks settings"
  [instance input]
  (let [activation-nodes (calculate/forward-propagate input
                                                      @(:thetas (:states instance))
                                                      (:sigmoid-fn instance))]
    (first (rseq activation-nodes))))


(defn- randomize-theta
  [input-nodes output-nodes]
  (let [theta (m/zero-array [output-nodes (inc input-nodes)])
        epsilon (/ (Math/sqrt 6) (Math/sqrt (+ input-nodes (inc output-nodes))))]
    (m/emap (fn [_] (- (* (rand) 2 epsilon) epsilon)) theta)))

(defn randomize-thetas
  "Create a randomize thetas for initial values

   It will the following formula

   `randomize(L0, L1) * 2 * epsilon - epsilon`

   Where L0 and L1 are the number of nodes adjacent to theta
   (e.g. input-node and hidden-layer-1-nodes, hidden-layer-1-nodes and output-nodes)

   Epsilon will be calculated using the following formula

   `sqrt(6) / sqrt(L0 + L1)`

   hidden-layers-nodes will be a vector of integers (number of nodes per hidden layer)
   "
  [input-nodes hidden-layers-nodes output-nodes]
  (when (and input-nodes output-nodes)
    (let [all-nodes (flatten [input-nodes hidden-layers-nodes output-nodes])]
      (loop [[first-nodes next-nodes & remaining-nodes] all-nodes
             thetas []]
        (let [thetas (conj thetas (randomize-theta first-nodes next-nodes))]
          (if (nil? remaining-nodes)
            thetas
            (recur (flatten [next-nodes remaining-nodes])
                   thetas)))))))

(defn new-instance
  "Creates new instance of neural networks.

   Options will be a hash map of

   ```
   {:regularization-rate value
    :activation-fn function
    :optimizer optimizer}
   ```

   Thetas would be the vector of initial weights matrices between each layer. To create a single
   hidden layer, Thetas would be a vector of two weight matrices.

   If optimizer is not specified, by default it will use gradient descent optimizer with the
   following settings:

   * initial learning rate of 8
   * learning rate update of 0.5
   * single stopping condition of 100 iterations

   Stopping conditions is a vector of stopping condition functions used by the optimizer which in
   turn used by neural networks training function.

   If multiple stopping conditions are provided, it will be treated as *OR* meaning as long as one
   of the condition is satisfied, training will be stopped (i.e. optimizer is finished)

   Returns a hashmap
   ```
   {
     :input input-matrix
     :output output-matrix
     :regularization-rate value (default is 0)
     :sigmoid-fn function (default is standard logistic)
     :optimizer optimizer function (default is gradient-descent with the default options)
     :states {
               :thetas [theta-matrix-1, theta-matrix-2, ...]
               :iteration (atom 0)
               :error (atom nil)
             }
   }
   ```"
  ([input thetas output]
   (new-instance input thetas output {}))
  ([input thetas output options]
   (let [merged-options (merge
                          {:regularization-rate 0
                           :sigmoid-fn          (standard-logistic)
                           :cost-fn 1 ;FIXME
                           :optimizer           (gradient-descent 8 0.5 [(max-iterations 100)])}
                          options)]
     {:input               input
      :output              output
      :regularization-rate (:regularization-rate merged-options)
      :sigmoid-fn          (:sigmoid-fn merged-options)
      :optimizer           (:optimizer merged-options)
      :states              {:thetas    (atom thetas)
                            :iteration (atom 0)
                            :error     (atom nil)}})))
