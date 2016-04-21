(ns neuralnetworks.sigmoid-fn-test
  (:require [clojure.test :refer :all]
            [neuralnetworks.utils :refer :all]
            [neuralnetworks.sigmoid-fn :as sigmoid-fn]
            [clojure.core.matrix :as m]))

(deftest test-standard-logistic
  (let [instance (sigmoid-fn/standard-logistic)]
    (are [x y] (approx x y)
               1 (sigmoid-fn/f instance 50)
               0 (sigmoid-fn/f instance -40)
               0.5 (sigmoid-fn/f instance 0))))

(deftest test-hyperbolic-tangent
  (let [instance (sigmoid-fn/hyperbolic-tangent)]
    (are [x y] (approx x y)
               1 (sigmoid-fn/f instance 50)
               0 (sigmoid-fn/f instance 0)
               -1 (sigmoid-fn/f instance -50))))

(deftest test-standard-logistic-derivative
  (let [instance (sigmoid-fn/standard-logistic)
        activated-nodes (m/array [0.8 0.1 0.5])]
    (is (m/equals (m/array [0.16 0.09 0.25]) (sigmoid-fn/f' instance activated-nodes) 1e-6))))

(deftest test-hyperbolic-tangent-derivative
  (let [instance (sigmoid-fn/hyperbolic-tangent)
        activated-nodes (m/array [0.8 0.1 0.5 -0.8])]
    (is (m/equals (m/array [0.36 0.99 0.75 0.36]) (sigmoid-fn/f' instance activated-nodes) 1e-6))))
