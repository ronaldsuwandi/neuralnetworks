(ns neuralnetworks.error-fn-test
  (:require [clojure.test :refer :all]
            [clojure.core.matrix :as m]
            [neuralnetworks.sigmoid-fn :as sigmoid-fn]
            [neuralnetworks.error-fn :as error-fn]
            [neuralnetworks.utils :refer [approx]]
            [neuralnetworks.calculate :as calculate]))

(deftest test-regularization-cost
  (let [theta1 (m/array [[0.10000 0.30000 0.50000]
                         [0.20000 0.40000 0.60000]])
        theta2 (m/array [[0.70000 1.10000 1.50000]
                         [0.80000 1.20000 1.60000]
                         [0.90000 1.30000 1.70000]
                         [1.00000 1.40000 1.80000]])
        expected 12.067]
    (is (approx expected (error-fn/regularization-cost 4 3 [theta1 theta2]) 0.001))))

(deftest test-cross-entropy
  (let [sigmoid (sigmoid-fn/standard-logistic)
        input (m/array [[0.54030 -0.41615]
                        [-0.98999 -0.65364]
                        [0.28366 0.96017]])
        theta1 (m/array [[0.10000 0.30000 0.50000]
                         [0.20000 0.40000 0.60000]])
        theta2 (m/array [[0.70000 1.10000 1.50000]
                         [0.80000 1.20000 1.60000]
                         [0.90000 1.30000 1.70000]
                         [1.00000 1.40000 1.80000]])
        all-thetas [theta1 theta2]
        output (m/array [[0 0 0 1]
                         [0 1 0 0]
                         [0 0 1 0]])
        activation-nodes (calculate/forward-propagate input all-thetas sigmoid)]
    (is (approx 7.406969 (error-fn/cross-entropy input all-thetas activation-nodes output 0)))
    (is (approx 19.473636 (error-fn/cross-entropy input all-thetas activation-nodes output 4)))))

(deftest test-mean-squared-error
  (let [sigmoid (sigmoid-fn/hyperbolic-tangent)
        input (m/array [[0.54030 -0.41615]
                        [-0.98999 -0.65364]
                        [0.28366 0.96017]])
        theta1 (m/array [[0.10000 0.30000 0.50000]
                         [0.20000 0.40000 0.60000]])
        theta2 (m/array [[0.70000 1.10000 1.50000]
                         [0.80000 1.20000 1.60000]
                         [0.90000 1.30000 1.70000]
                         [1.00000 1.40000 1.80000]])
        all-thetas [theta1 theta2]
        output (m/array [[0 0 0 1]
                         [0 1 0 0]
                         [0 0 1 0]])
        activation-nodes (calculate/forward-propagate input all-thetas sigmoid)]
    (is (approx 2.070877 (error-fn/mean-squared-error input all-thetas activation-nodes output 0)))
    (is (approx 14.137544 (error-fn/mean-squared-error input
                                                       all-thetas
                                                       activation-nodes
                                                       output
                                                       4)))))
