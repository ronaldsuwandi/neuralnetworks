(ns neuralnetworks.optimizer.stopping-conditions)

(defn min-error-rate
  "Stops when neural networks instance error is less than specified rate."
  [^double error-rate]
  (fn [optimizer]
    (< (:error optimizer) error-rate)))

(defn max-iterations
  "Stops when neural networks iteration count reaches the specified value"
  [^long iterations]
  (fn [optimizer]
    (>= (:iteration optimizer) iterations)))
