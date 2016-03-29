(ns neuralnetworks.activation-fn)

(defn sigmoid
  ^double [^double x]
  (double (/ 1 (+ 1 (Math/exp (- x))))))

(defn binary
  ^double [^double x]
  (if (< x 0.5) 0.0 1.0))
