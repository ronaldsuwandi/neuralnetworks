(ns neuralnetworks.sigmoid-fn-test
  (:require [clojure.test :refer :all]
            [neuralnetworks.utils :refer :all]
            [neuralnetworks.sigmoid-fn :as sigmoid-fn]))

(deftest test-standard-logistic
  (are [x y] (approx x y)
             1 (sigmoid-fn/standard-logistic 50)
             0 (sigmoid-fn/standard-logistic -40)
             0.5 (sigmoid-fn/standard-logistic 0)))

(deftest test-hyperbolic-tangent
  (are [x y] (approx x y)
             1.7159 (sigmoid-fn/hyperbolic-tangent 50)
             0 (sigmoid-fn/hyperbolic-tangent 0)
             -1.7159 (sigmoid-fn/hyperbolic-tangent -50)))
