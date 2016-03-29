(ns neuralnetworks.optimizer.gradient-descent-test
  (:require [clojure.test :refer :all]
            [clojure.core.matrix :as m]
            [neuralnetworks.optimizer :as optimizer]
            [neuralnetworks.optimizer.gradient-descent :as gd]
            [neuralnetworks.optimizer.stopping-conditions :refer [max-iterations]]))

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
        gradient-descent (gd/gradient-descent 1 [(max-iterations 2)])]
    (is (thrown? Exception (optimizer/optimize gradient-descent cost-fn [1])))))

(deftest test-gradient-descent-optimizer
  (let [cost-fn (fn [theta] {:cost (m/esum theta) :gradients (- (m/esum theta) 5) :theta theta})
        gradient-descent (gd/gradient-descent 1 [(max-iterations 3)])]
    (is (= {:iteration 3 :error nil :thetas [5]}
           (optimizer/optimize gradient-descent cost-fn [10 10])))))

