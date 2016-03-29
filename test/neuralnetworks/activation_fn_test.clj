(ns neuralnetworks.activation-fn-test
  (:require [clojure.test :refer :all]
            [neuralnetworks.utils :refer :all]
            [neuralnetworks.activation-fn :as activation-fn]))

(deftest test-sigmoid
  (are [x y] (approx x y)
             1 (activation-fn/sigmoid 50)
             0 (activation-fn/sigmoid -40)
             0.5 (activation-fn/sigmoid 0)))

(deftest test-binary
  (are [x y] (approx x y)
             1 (activation-fn/binary 1)
             1 (activation-fn/binary 10)
             0 (activation-fn/binary 0)
             0 (activation-fn/binary -10)))
