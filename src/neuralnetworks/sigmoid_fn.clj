(ns neuralnetworks.sigmoid-fn
  (:require [clojure.core.matrix :as m]))

(defprotocol Sigmoid
  "Sigmoid function protocol"
  (f ^double [^double x] "Calculate the sigmoid function for the given x parameter")
  (f' [activated-nodes]
    "Calculate the derivative of the sigmoid function for the given activated nodes (nodes with
     sigmoid function applied to it). This is done for efficiency so we don't have to keep performing
     sigmoid function over and over again"))

(defrecord HyperbolicTangent []
  Sigmoid
  (f ^double [^double x]
    (double (* 1.7159 (Math/tanh (* x (/ 2 3))))))
  (f' [activated-nodes]
    (m/sub 1 (m/pow activated-nodes 2))))

(defrecord StandardLogistic []
  Sigmoid
  (f ^double [^double x]
    (double (/ 1 (+ 1 (Math/exp (- x))))))
  (f' [activated-nodes]
    (m/mul activated-nodes (m/sub 1 activated-nodes))))

(defn hyperbolic-tangent
  "Returns new instance of HyperbolicTangent sigmoid function.

   It's an optimized version of hyperbolic tangent function where it uses the following formula:
   `1.7159 * tanh(2/3 * x)` [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)

   Yields output between -1.7159 and 1.7159

   Derivative is `1-f(x)^2`"
  (->HyperbolicTangent))

(defn standard-logistic
  "Returns new instance of StandardLogistic sigmoid function which uses the following formula:
   `1/(1+e^-x)`

   Yields output between 0 and 1

   Derivative is `f(x)+(1-f(x))`"
  (->StandardLogistic))

(alter-meta! #'->HyperbolicTangent assoc :no-doc true)
(alter-meta! #'->StandardLogistic assoc :no-doc true)
