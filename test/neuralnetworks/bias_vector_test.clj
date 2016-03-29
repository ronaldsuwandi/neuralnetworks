(ns neuralnetworks.bias-vector-test
  (:require [clojure.test :refer :all]
            [neuralnetworks.utils :refer :all]
            [neuralnetworks.bias-vector :as bias]
            [clojure.core.matrix :as m]))

(deftest test-append
  (let [input (m/array [[1 2]
                        [3 4]])
        expected (m/array [[1 1 2]
                           [1 3 4]])]
    (is (m/equals expected (bias/append input) 1e-6))))

(deftest test-delete
  (let [input (m/array [[1 1 2]
                        [1 3 4]])
        expected (m/array [[1 2]
                           [3 4]])]
    (is (m/equals expected (bias/delete input) 1e-6))))
