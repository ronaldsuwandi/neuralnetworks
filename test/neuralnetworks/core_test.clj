(ns neuralnetworks.core-test
  (:require [clojure.test :refer :all]
            [neuralnetworks.core :as nn]
            [neuralnetworks.optimizer.stopping-conditions :refer [max-iterations]]
            [neuralnetworks.optimizer.gradient-descent :as gd]
            [clojure.core.matrix :as m]))

(deftest test-train-and
  (let [input (m/array [[0 0]
                        [0 1]
                        [1 0]
                        [1 1]])
        ; thetas values are generated from `(nn/randomize-thetas 2 [3] 1)`
        theta-input-hidden-layer (m/array [[-0.453452 0.510620 -0.773563]
                                           [-0.942836 0.027642 0.182300]
                                           [-0.534591 0.786677 0.756434]])
        theta-hidden-layer-output (m/array [[0.453169 -1.037344 -0.837474 0.985438]])
        thetas [theta-input-hidden-layer theta-hidden-layer-output]
        output (m/array [[0] [0] [0] [1]])
        instance (nn/new-instance input thetas output {})]
    (nn/train! instance)
    (is (m/equals output (nn/predict instance input) 0.1))))

(deftest test-train-xor
  (let [input (m/array [[0 0]
                        [0 1]
                        [1 0]
                        [1 1]])
        ; thetas values are generated from `(nn/randomize-thetas 2 [3] 1)`
        theta-input-hidden-layer (m/array [[-0.453452 0.510620 -0.773563]
                                           [-0.942836 0.027642 0.182300]
                                           [-0.534591 0.786677 0.756434]])
        theta-hidden-layer-output (m/array [[0.453169 -1.037344 -0.837474 0.985438]])
        thetas [theta-input-hidden-layer theta-hidden-layer-output]
        output (m/array [[0] [1] [1] [0]])
        instance (nn/new-instance input thetas output {})]
    (nn/train! instance)
    (is (m/equals output (nn/predict instance input) 0.1))))

(deftest test-randomize-theta-dimensions
  (let [thetas-no-hidden-layer (nn/randomize-thetas 2 [] 1)
        thetas-one-hidden-layer (nn/randomize-thetas 2 [3] 1)
        thetas-two-hidden-layers (nn/randomize-thetas 2 [1 1] 1)]
    (is (= [[1 3]] (map m/shape thetas-no-hidden-layer)))
    (is (= [[3 3] [1 4]] (map m/shape thetas-one-hidden-layer)))
    (is (= [[1 3] [1 2] [1 2]] (map m/shape thetas-two-hidden-layers)))))
