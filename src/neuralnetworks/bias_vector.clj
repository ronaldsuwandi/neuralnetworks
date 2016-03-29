(ns neuralnetworks.bias-vector
  (:require [clojure.core.matrix :as m]))

(defn append
  "Append bias vector into the first column of the given matrix"
  [nodes]
  (let [bias-vector (m/broadcast 1 [(m/row-count nodes) 1])]
    (m/join-along 1 bias-vector nodes)))

(defn delete
  "Remove bias vector from the first column of the given matrix"
  [nodes]
  (m/submatrix nodes 1 [1 (dec (m/column-count nodes))]))
