(ns neuralnetworks.core
  (:require [neuralnetworks.sigmoid-fn :as sigmoid]
            [neuralnetworks.error-fn :as error]
            [neuralnetworks.calculate :as calculate]
            [neuralnetworks.optimizer :as optimizer]
            [neuralnetworks.optimizer.gradient-descent :refer [gradient-descent]]
            [neuralnetworks.optimizer.stopping-conditions :refer [max-iterations]]
            [clojure.core.matrix :as m]
            [taoensso.timbre :as log]))

(defn train!
  "Train the neural networks. This will update the thetas/weights

   Stopping conditions is a vector of stopping condition functions used by the optimizer which in
   turn used by neural networks training function.

   If multiple stopping conditions are provided, it will be treated as *OR* meaning as long as one
   of the condition is satisfied, training will be stopped (i.e. optimizer is finished)"
  [instance stopping-conditions]
  (log/debug "Training started")
  (let [start (System/currentTimeMillis)
        states (:states instance)
        thetas @(:thetas states)
        thetas-dimensions (mapv m/shape thetas)
        cost-fn (calculate/cost-fn (:input instance)
                                   (:output instance)
                                   (:error-fn instance)
                                   (:sigmoid-fn instance)
                                   (:regularization-rate instance)
                                   thetas-dimensions)
        optimized (optimizer/optimize (:optimizer instance)
                                      cost-fn
                                      (m/as-vector thetas)
                                      stopping-conditions)]
    (reset! (:iteration states) (:iteration optimized))
    (reset! (:thetas states) (calculate/reshape-thetas (:thetas optimized) thetas-dimensions))
    (log/debugf "Training completed in %dms" (- (System/currentTimeMillis) start))
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

   Problem-type accepts either `:classification` or `:regression`. Problem-type determines the
   default sigmoid and error function

   For `classification` it will use

   ```
   {:sigmoid-fn standard-logistic
    :error-fn cross-entropy}
   ```

   for the options. Cross-entropy is more suitable because it penalizes misclassification

   Otherwise, for `:regression` it will use

   ```
   {:sigmoid-fn hyperbolic-tangent
    :error-fn mean-squared-error}
   ```

   Mean squared error is best suited for regression (curve-fitting) problem

   Options will be a hash map of

   ```
   {:regularization-rate value
    :activation-fn function
    :sigmoid-fn function       ; optional, if you want to customize/override sigmoid function
    :error-fn function         ; optional, if you want to customize/override error function
    :optimizer optimizer}
   ```

   Thetas would be the vector of initial weights matrices between each layer. To create a single
   hidden layer, Thetas would be a vector of two weight matrices.

   If optimizer is not specified, by default it will use gradient descent optimizer with the
   following settings:

   * initial learning rate of 4
   * learning rate update of 0.5

   *Note* it is important to always normalize the input and output nodes for better performance

   Returns a hashmap
   ```
   {
     :input input-matrix
     :output output-matrix
     :regularization-rate value ; default is 0
     :sigmoid-fn function       ; default is standard logistic for classification, hyperbolic
                                ; tangent for regression
     :errror-fn function        ; default is cross-entropy for classification, mean squared error
                                ; for regression
     :optimizer function        ; default is gradient-descent with the default options
     :states {
               :thetas [theta-matrix-1, theta-matrix-2, ...]
               :iteration (atom 0)
               :error (atom nil)
             }
   }
   ```"
  ([input thetas output problem-type]
   {:pre [(contains? #{:classification :regression} problem-type)]}
   (new-instance input thetas output problem-type {}))
  ([input thetas output problem-type options]
   {:pre [(contains? #{:classification :regression} problem-type)]}
   (let [default-options-for-type {:classification {:sigmoid-fn (sigmoid/standard-logistic)
                                                    :error-fn   error/cross-entropy}
                                   :regression     {:sigmoid-fn (sigmoid/hyperbolic-tangent)
                                                    :error-fn   error/mean-squared-error}}
         default-options {:regularization-rate 0
                          :optimizer           (gradient-descent 4 0.5)}
         merged-options (merge default-options (get default-options-for-type problem-type) options)]
     {:input               input
      :output              output
      :regularization-rate (:regularization-rate merged-options)
      :sigmoid-fn          (:sigmoid-fn merged-options)
      :error-fn            (:error-fn merged-options)
      :optimizer           (:optimizer merged-options)
      :states              {:thetas    (atom thetas)
                            :iteration (atom 0)}})))
