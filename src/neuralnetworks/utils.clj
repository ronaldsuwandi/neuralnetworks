(ns neuralnetworks.utils)

(defn approx
  "Check if the two numbers are approximately equal (if the two numbers differences are less than
  specified epsilon or using default one (1e-6)"
  ([^double no1 ^double no2 ^double epsilon]
   (< (Math/abs (- no1 no2)) epsilon))
  ([^double no1 ^double no2]
   (approx no1 no2 1e-6)))
