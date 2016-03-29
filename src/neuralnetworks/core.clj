(ns neuralnetworks.core
  (:require [neuralnetworks.activation-fn :refer [sigmoid]]
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
                                   (:activation-fn instance)
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
                                                      (:activation-fn instance))]
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
   {:learning-rate value
    :regularization-rate value
    :activation-fn function
    :stopping-conditions [functions]
    :optimizer optimizer}

   Thetas would be the vector of initial weights matrices between each layer. To create a single
   hidden layer, Thetas would be a vector of two weight matrices.

   Stopping conditions is a vector of stopping condition functions. By default training will be
   finished once it reaches 100 iterations (max-iteration stopping condition). If multiple stopping
   conditions are provided, it will be treated as *OR* meaning as long as one of the condition is
   satisfied, training will be stopped

   Returns a hashmap
   ```
   {
     :input input-matrix
     :output output-matrix
     :regularization-rate value (default is 0)
     :learning-rate value (default is 1)
     :stopping-conditions [function-1, function-2, ...] (default is 100 iterations)
     :activation-fn function (default is sigmoid)
     :optimizer optimizer function (default is gradient-descent)
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
   (let [merged-options (merge {:regularization-rate 0
                                :learning-rate       1
                                :activation-fn       sigmoid
                                :stopping-conditions [(max-iterations 100)]}
                               options)
         merged-options (merge {:optimizer (gradient-descent (:learning-rate merged-options)
                                                             (:stopping-conditions merged-options))}
                               merged-options)]
     {:input               input
      :output              output
      :regularization-rate (:regularization-rate merged-options)
      :learning-rate       (:learning-rate merged-options)
      :stopping-conditions (:stopping-conditions merged-options)
      :activation-fn       (:activation-fn merged-options)
      :optimizer           (:optimizer merged-options)
      :states              {:thetas    (atom thetas)
                            :iteration (atom 0)
                            :error     (atom nil)}})))
