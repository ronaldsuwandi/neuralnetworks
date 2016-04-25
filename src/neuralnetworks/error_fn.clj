(ns neuralnetworks.error-fn
  (:require [clojure.core.matrix :as m]))

(defn regularization-cost
  "thetas is a vector of matrices"
  ^double [lambda input-count thetas]
  (let [lambda-constant (/ lambda (* 2 input-count))
        thetas-without-bias-column (map #(m/select % :all :rest) thetas)
        sum-of-thetas-squared (->> (map #(m/pow % 2) thetas-without-bias-column)
                                   (map m/esum)
                                   (reduce +))]
    (* sum-of-thetas-squared lambda-constant)))

(defn cross-entropy
  "Suitable for classification problem. Note this cost function will not work well with hyperbolic
   tangent sigmoid function due to the presence of negative number"
  [input thetas activation-nodes output lambda]
  {:pre [(seq thetas) (seq activation-nodes)]}
  (let [regularization-cost (regularization-cost lambda (m/row-count input) thetas)
        last-activation-nodes (get activation-nodes (dec (count activation-nodes)))
        last-activation-nodes-count (m/row-count last-activation-nodes)
        inner-cost-values (m/add
                            (m/mul output (m/log last-activation-nodes))
                            (m/mul (m/sub 1 output)
                                   (m/log (m/sub 1 last-activation-nodes))))]
    (+ (/ (m/esum inner-cost-values) (- last-activation-nodes-count))
       regularization-cost)))

(defn mean-squared-error
  "Suitable for linear regression problem (curve fitting)"
  [input thetas activation-nodes output lambda]
  {:pre [(seq thetas) (seq activation-nodes)]}
  (let [regularization-cost (regularization-cost lambda (m/row-count input) thetas)
        last-activation-nodes (get activation-nodes (dec (count activation-nodes)))
        last-activation-nodes-count (m/row-count last-activation-nodes)
        difference-squared (m/pow (m/sub last-activation-nodes output) 2)]
    (double (+ (/ (m/esum difference-squared) (* 2 last-activation-nodes-count))
               regularization-cost))))
