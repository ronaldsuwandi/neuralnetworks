(ns neuralnetworks.calculate-test
  (:require [clojure.test :refer :all]
            [neuralnetworks.calculate :as calculate]
            [neuralnetworks.sigmoid-fn :as sigmoid-fn]
            [neuralnetworks.error-fn :as error-fn]
            [neuralnetworks.utils :refer :all]
            [clojure.core.matrix :as m]))

(deftest test-output-nodes
  (let [input (m/array [[1 0 0]
                        [1 0 1]
                        [1 1 0]
                        [1 1 1]])
        weights (m/array [-60 40 40])
        expected (m/array [0 0 0 1])
        output (calculate/output-nodes input weights (sigmoid-fn/standard-logistic))]
    (is (m/equals expected output 1e-6))))

(deftest test-forward-propagate
  (let [input (m/array [[0.54030 -0.41615]
                        [-0.98999 -0.65364]
                        [0.28366 0.96017]])
        theta1 (m/array [[0.10000 0.30000 0.50000]
                         [0.20000 0.40000 0.60000]])
        theta2 (m/array [[0.70000 1.10000 1.50000]
                         [0.80000 1.20000 1.60000]
                         [0.90000 1.30000 1.70000]
                         [1.00000 1.40000 1.80000]])
        all-thetas [theta1 theta2]
        first-activation-nodes (m/array [[0.513500 0.541511]
                                         [0.371960 0.357052]
                                         [0.660423 0.708800]])
        last-activation-nodes (m/array [[0.888659 0.907427 0.923304 0.936649]
                                        [0.838178 0.860282 0.879799 0.896917]
                                        [0.923414 0.938577 0.950898 0.960850]])
        result (calculate/forward-propagate input all-thetas (sigmoid-fn/standard-logistic))]

    (is (m/equals first-activation-nodes (get result 0) 1e-6))
    (is (m/equals last-activation-nodes (get result 1) 1e-6))))

(deftest test-cost
  (let [sigmoid (sigmoid-fn/standard-logistic)
        cost error-fn/cross-entropy
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
        thetas-dimensions (mapv m/shape all-thetas)
        thetas (m/as-vector all-thetas)
        output (m/array [[0 0 0 1]
                         [0 1 0 0]
                         [0 0 1 0]])
        cost-fn-no-regularization (calculate/cost-fn input output cost sigmoid 0 thetas-dimensions)
        cost-fn-regularization (calculate/cost-fn input output cost sigmoid 4 thetas-dimensions)]
    (is (approx 7.406969 (:cost (cost-fn-no-regularization thetas))))
    (is (approx 19.473636 (:cost (cost-fn-regularization thetas))))))

(deftest test-cost-args
  (let [sigmoid (sigmoid-fn/standard-logistic)
        cost error-fn/cross-entropy
        input (m/array [[0 0]
                        [1 1]
                        [1 1]])
        theta (m/array [[1 2 3]
                        [4 5 6]])
        thetas-dimensions (mapv m/shape [theta])
        output (m/array [[0 0]
                         [0 1]
                         [0 0]])
        cost-fn (calculate/cost-fn input output cost sigmoid 0 thetas-dimensions)]
    (is (not (contains? (cost-fn (m/as-vector theta) :skip-gradient) :gradients)))
    (is (contains? (cost-fn (m/as-vector theta)) :gradients))))

(deftest test-delta
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
        activation-nodes (calculate/forward-propagate input all-thetas sigmoid)
        delta2 (m/array [[2.650251 1.377940 1.435009]
                         [1.706287 1.033853 1.106760]
                         [1.754003 0.768939 0.779311]
                         [1.794417 0.935655 0.966993]])
        delta1 (m/array [[2.298415 -0.082621 -0.074786]
                         [2.939692 -0.107535 -0.161585]])
        deltas (calculate/delta input all-thetas activation-nodes output sigmoid)]
    (is (m/equals delta1 (get deltas 0) 1e-6))
    (is (m/equals delta2 (get deltas 1) 1e-6))))

(deftest test-theta-gradient
  (let [delta2 (m/array [[2.650251 1.377940 1.435009]
                         [1.706287 1.033853 1.106760]
                         [1.754003 0.768939 0.779311]
                         [1.794417 0.935655 0.966993]])
        delta1 (m/array [[2.298415 -0.082621 -0.074786]
                         [2.939692 -0.107535 -0.161585]])
        deltas [delta1 delta2]
        theta1 (m/array [[0.10000 0.30000 0.50000]
                         [0.20000 0.40000 0.60000]])
        theta2 (m/array [[0.70000 1.10000 1.50000]
                         [0.80000 1.20000 1.60000]
                         [0.90000 1.30000 1.70000]
                         [1.00000 1.40000 1.80000]])
        all-thetas [theta1 theta2]
        input-count 3
        theta-gradient1-no-lambda (m/array [[0.766138 -0.027540 -0.024928]
                                            [0.979897 -0.035845 -0.053861]])
        theta-gradient2-no-lambda (m/array [[0.883416 0.459313 0.478336]
                                            [0.568762 0.344617 0.368919]
                                            [0.584667 0.256313 0.259770]
                                            [0.598139 0.311884 0.322331]])
        theta-gradient1-lambda (m/array [[0.766138 0.372459 0.641737]
                                         [0.979897 0.497488 0.746138]])
        theta-gradient2-lambda (m/array [[0.883416 1.925979 2.478336]
                                         [0.568762 1.944617 2.502253]
                                         [0.584667 1.989646 2.526436]
                                         [0.598139 2.178551 2.722330]])
        theta-gradients-no-lambda (calculate/theta-gradient deltas all-thetas 0 input-count)
        theta-gradients-lambda (calculate/theta-gradient deltas all-thetas 4 input-count)]

    (is (m/equals theta-gradient1-no-lambda (get theta-gradients-no-lambda 0) 1e-6))
    (is (m/equals theta-gradient2-no-lambda (get theta-gradients-no-lambda 1) 1e-6))
    (is (m/equals theta-gradient1-lambda (get theta-gradients-lambda 0) 1e-6))
    (is (m/equals theta-gradient2-lambda (get theta-gradients-lambda 1) 1e-6))))

(deftest test-reshape-thetas
  (let [original-thetas [(m/array [[0 1 2] [3 4 5]])
                         (m/array [[7 8] [9 10] [11 12]])]
        thetas-dimensions (mapv m/shape original-thetas)
        thetas (m/to-vector original-thetas)]
    (is (m/equals original-thetas (calculate/reshape-thetas thetas thetas-dimensions)))))
