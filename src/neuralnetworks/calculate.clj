(ns neuralnetworks.calculate
  (:require [clojure.core.matrix :as m]
            [neuralnetworks.sigmoid-fn :as sigmoid]
            [neuralnetworks.bias-vector :as bias]
            [neuralnetworks.utils :refer :all]))

(defn output-nodes
  [input theta sigmoid-fn]
  (m/emap #(sigmoid/f sigmoid-fn %) (m/mmul input theta)))

(defn forward-propagate
  "Will return list of activation nodes for each theta"
  [input thetas sigmoid-fn]
  (when (seq thetas)
    (loop [[theta & remaining-thetas] thetas
           input-with-bias (bias/append input)
           all-activation-nodes []]
      (let [activation-nodes (output-nodes input-with-bias (m/transpose theta) sigmoid-fn)
            all-activation-nodes (conj all-activation-nodes activation-nodes)]
        (if-not (empty? remaining-thetas)
          (let [activation-nodes-with-bias (bias/append activation-nodes)]
            (recur remaining-thetas
                   activation-nodes-with-bias
                   all-activation-nodes))
          all-activation-nodes)))))

(defn delta
  "Calculate the 'error' (delta) from the expected input/output for the given thetas.
   This will also calculate the delta for each hidden layers."
  [input thetas activation-nodes output sigmoid-fn]
  (when (and (seq thetas) (seq activation-nodes))
    ; error/delta is calculated backwards, thus the reverse order of activation nodes and thetas
    (let [[activation-node & remaining-activation-nodes] (rseq activation-nodes)
          error (m/sub activation-node output)]
      (loop [last-error error
             [activation-node & remaining-activation-nodes] remaining-activation-nodes
             [theta & remaining-thetas] (rseq thetas)
             deltas []]
        (if-not (empty? remaining-thetas)
          (let [error (m/mul (bias/delete (m/mmul last-error theta))
                             (sigmoid/f' sigmoid-fn activation-node))
                delta (m/mmul (m/transpose last-error) (bias/append activation-node))
                deltas (conj deltas delta)]
            (recur error remaining-activation-nodes remaining-thetas deltas))
          ; reverse the order of delta
          (let [delta (m/mmul (m/transpose last-error) (bias/append input))
                deltas (conj deltas delta)]
            (vec (rseq deltas))))))))

(defn- theta-gradient-regularization
  [lambda-constant theta theta-gradient]
  (let [theta-gradient-bias (m/submatrix theta-gradient 1 [0 1])
        theta-gradient-without-bias (bias/delete theta-gradient)
        theta-without-bias (bias/delete theta)]
    (m/join-along 1 theta-gradient-bias
                  (m/add theta-gradient-without-bias (m/mul theta-without-bias lambda-constant)))))

(defn theta-gradient
  [deltas thetas lambda input-count]
  (let [lambda-constant (/ lambda input-count)]
    (->> deltas
         (map #(m/div % input-count))
         (mapv (partial theta-gradient-regularization lambda-constant) thetas))))

(defn reshape-thetas
  "Reshapes theta vectors based on the given thetas-dimensions where thetas-dimensions are vector of
   matrices dimensions"
  [thetas thetas-dimensions]
  (loop [thetas-vector thetas
         [dimension & remaining-dimensions] thetas-dimensions
         reshaped-thetas []]
    (let [subvector-start (reduce * dimension)
          subvector-length (- (m/row-count thetas-vector) subvector-start)]
      (if dimension
        (recur (m/subvector thetas-vector subvector-start subvector-length)
               remaining-dimensions
               (conj reshaped-thetas (m/array (m/reshape thetas-vector dimension))))
        reshaped-thetas))))

(defn cost-fn
  [input output error-fn sigmoid-fn lambda thetas-dimensions]
  (fn [thetas & args]
    (let [reshaped-thetas (reshape-thetas thetas thetas-dimensions)
          activation-nodes (forward-propagate input reshaped-thetas sigmoid-fn)
          error-value (error-fn input reshaped-thetas activation-nodes output lambda)
          result {:cost error-value}]
      (if (contains? (set args) :skip-gradient)
        result
        (let [deltas (delta input reshaped-thetas activation-nodes output sigmoid-fn)
              theta-gradients (theta-gradient deltas reshaped-thetas lambda (m/row-count input))]
          (assoc result :gradients theta-gradients))))))
