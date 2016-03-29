(ns neuralnetworks.calculate
  (:require [clojure.core.matrix :as m]
            [neuralnetworks.bias-vector :as bias]
            [neuralnetworks.utils :refer :all]))

(defn output-nodes
  [input theta activation-fn]
  (m/emap activation-fn (m/mmul input theta)))

(defn regularization-cost
  "thetas is a vector of matrices"
  ^double [lambda input-count thetas]
  (let [lambda-constant (/ lambda (* 2 input-count))
        thetas-without-bias-column (map #(m/select % :all :rest) thetas)
        sum-of-thetas-squared (->> (map #(m/pow % 2) thetas-without-bias-column)
                                   (map m/esum)
                                   (reduce +))]
    (* sum-of-thetas-squared lambda-constant)))

(defn forward-propagate
  "Will return list of activation nodes for each theta"
  [input thetas activation-fn]
  (when (seq thetas)
    (loop [[theta & remaining-thetas] thetas
           input-with-bias (bias/append input)
           all-activation-nodes []]
      (let [activation-nodes (output-nodes input-with-bias (m/transpose theta) activation-fn)
            all-activation-nodes (conj all-activation-nodes activation-nodes)]
        (if-not (empty? remaining-thetas)
          (let [activation-nodes-with-bias (bias/append activation-nodes)]
            (recur remaining-thetas
                   activation-nodes-with-bias
                   all-activation-nodes))
          all-activation-nodes)))))

(defn cost
  [input thetas activation-nodes output lambda]
  (when (and (seq thetas) (seq activation-nodes))
    (let [regularization-cost (regularization-cost lambda (m/row-count input) thetas)
          last-activation-nodes (get activation-nodes (dec (count activation-nodes)))]
      (let [last-activation-nodes-count (m/row-count last-activation-nodes)
            inner-cost-values (m/add
                                (m/mul output (m/log last-activation-nodes))
                                (m/mul (m/sub 1 output)
                                       (m/log (m/sub 1 last-activation-nodes))))]
        (+ (/ (m/esum inner-cost-values) (- last-activation-nodes-count))
           regularization-cost)))))

(defn derivative
  "Returns the derivative for the given activated node (after applying sigmoid function)"
  [activated-nodes]
  (m/mul activated-nodes (m/sub 1 activated-nodes)))

(defn delta
  "Calculate the 'error' (delta) from the expected input/output for the given thetas.
   This will also calculate the delta for each hidden layers."
  [input thetas activation-nodes output]
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
                             (derivative activation-node))
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

(defn error
  ^double [expected output]
  (let [data-count (m/row-count output)
        difference-squared (m/pow (m/sub output expected) 2)]
    (double (/ (m/esum difference-squared) (* 2 data-count)))))

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
  [input output activation-fn lambda thetas-dimensions]
  (fn [thetas]
    (let [reshaped-thetas (reshape-thetas thetas thetas-dimensions)
          activation-nodes (forward-propagate input reshaped-thetas activation-fn)
          cost-value (cost input reshaped-thetas activation-nodes output lambda)
          deltas (delta input reshaped-thetas activation-nodes output)
          theta-gradients (theta-gradient deltas reshaped-thetas lambda (m/row-count input))
          error-value (error output (first (rseq activation-nodes)))]
      {:cost      cost-value
       :error     error-value
       :gradients theta-gradients})))
