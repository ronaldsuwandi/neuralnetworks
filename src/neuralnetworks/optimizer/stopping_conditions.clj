(ns neuralnetworks.optimizer.stopping-conditions)

(defn max-error
  "Stops when neural networks instance error is less than specified rate."
  [^double error]
  (fn [optimizer]
    (< (:error optimizer) error)))

(defn max-iterations
  "Stops when neural networks iteration count reaches the specified value"
  [^long iterations]
  (fn [optimizer]
    (>= (:iteration optimizer) iterations)))
