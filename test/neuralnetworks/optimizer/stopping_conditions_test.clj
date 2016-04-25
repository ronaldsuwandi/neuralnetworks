(ns neuralnetworks.optimizer.stopping-conditions-test
  (:require [clojure.test :refer :all]
            [neuralnetworks.optimizer.stopping-conditions :as s]))

(deftest test-error-rate
  (let [stopping-fn (s/max-error 0.01)]
    (is (not (stopping-fn {:error 0.5})))
    (is (stopping-fn {:error 0}))
    (is (stopping-fn {:error 1e-10}))))

(deftest test-max-iterations
  (let [stopping-fn (s/max-iterations 1000)]
    (is (not (stopping-fn {:iteration 5})))
    (is (stopping-fn {:iteration 1000}))
    (is (stopping-fn {:iteration 5000}))))
