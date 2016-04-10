(ns neuralnetworks.optimizer.gradient-descent-test
  (:require [clojure.test :refer :all]
            [clojure.core.matrix :as m]
            [neuralnetworks.optimizer :as optimizer]
            [neuralnetworks.optimizer.gradient-descent :as gd]
            [neuralnetworks.optimizer.stopping-conditions :refer [max-iterations]]
            [neuralnetworks.utils :refer [approx]]))

(deftest test-update-thetas
  (let [theta1 (m/array [[0.10000 0.30000 0.50000]
                         [0.20000 0.40000 0.60000]])
        theta2 (m/array [[0.70000 1.10000 1.50000]
                         [0.80000 1.20000 1.60000]
                         [0.90000 1.30000 1.70000]
                         [1.00000 1.40000 1.80000]])
        thetas-vector (m/as-vector [theta1 theta2])
        theta-gradient1 (m/array [[0.766138 -0.027540 -0.024928]
                                  [0.979897 -0.035845 -0.053861]])
        theta-gradient2 (m/array [[0.883416 0.459313 0.478336]
                                  [0.568762 0.344617 0.368919]
                                  [0.584667 0.256313 0.259770]
                                  [0.598139 0.311884 0.322331]])
        alpha 0.01
        theta-gradients-vector (m/as-vector [theta-gradient1 theta-gradient2])
        updated-theta1 (m/array [[0.092338 0.300275 0.500249]
                                 [0.190201 0.400358 0.600538]])
        updated-theta2 (m/array [[0.691165 1.095406 1.495216]
                                 [0.794312 1.196553 1.596310]
                                 [0.894153 1.297436 1.697402]
                                 [0.994018 1.396881 1.796776]])
        expected-updated-theta-vector (m/as-vector [updated-theta1 updated-theta2])
        updated-theta-vector (gd/update-thetas thetas-vector theta-gradients-vector alpha)]
    (is (m/equals expected-updated-theta-vector updated-theta-vector 1e-6))))

(deftest test-check-for-gradients
  (let [cost-fn (fn [_] {:cost 123})
        gradient-descent (gd/gradient-descent 1 1 [(max-iterations 2)])]
    (is (thrown? Exception (optimizer/optimize gradient-descent cost-fn [1])))))

(deftest test-gradient-descent-optimizer
  (let [cost-fn (fn [theta]
                  ; cost function is [x.^2 + y.^2]
                  ; gradient is the derivative [2x 2y]
                  (let [cost (m/esum (m/pow theta 2))]
                    {:cost      cost
                     :gradients (m/mul theta 2)
                     :error     (Math/abs (- 0 cost))
                     :theta     theta}))
        initial-theta [1000000 -1000000]
        gradient-descent (gd/gradient-descent 4 0.8 [(max-iterations 15)])
        gradient-descent-without-line-search (gd/gradient-descent 0.1 1 [(max-iterations 100)])
        result (optimizer/optimize gradient-descent cost-fn initial-theta)
        result-without-line-search (optimizer/optimize gradient-descent-without-line-search
                                                       cost-fn
                                                       initial-theta)]

    (is (approx 0 (:error result) 1e-12))
    (is (approx 0 (:error result-without-line-search) 1e-6))
    (is (< (:error result) (:error result-without-line-search)))))

(deftest test-line-search
  (let [cost-fn (fn [theta]
                  ; cost function is [x.^2 + y.^2]
                  ; gradient is the derivative [2x 2y]
                  (let [cost (m/esum (m/pow theta 2))]
                    {:cost      cost
                     :gradients (m/emap - (m/mul theta 2))
                     :error     (Math/abs (- 0 cost))
                     :theta     theta}))
        alpha-without-line-search (gd/line-search cost-fn 0.1 1 [10 -10] [20 -20])
        alpha-with-line-search (gd/line-search cost-fn 4 0.5 [10 -10] [20 -20])]
    (is (approx 0.1 alpha-without-line-search))
    (is (approx 0.5 alpha-with-line-search))))
