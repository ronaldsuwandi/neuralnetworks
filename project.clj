(defproject ronaldsuwandi/neuralnetworks "0.1.1-SNAPSHOT"
  :description "Neural networks library for Clojure. Built on top of core.matrix"
  :url "https://ronaldsuwandi.github.io/neuralnetworks/"
  :license {:name "MIT License"
            :url  "https://opensource.org/licenses/MIT"}
  :plugins [[lein-codox "0.9.4"]
            [lein-kibit "0.1.2"]
            [jonase/eastwood "0.2.3"]
            [lein-bikeshed "0.3.0"]]
  :codox {:metadata   {:doc/format :markdown}
          :source-uri "https://github.com/ronaldsuwandi/neuralnetworks/blob/master/{filepath}#L{line}"}
  :profiles {:test
             {:dependencies [[criterium/criterium "0.4.4"]]}}
  :aliases {"quality" ["do"
                       ["kibit"]
                       ["eastwood"]
                       ["bikeshed" "-m" "100"]]
            "check"   ["do"
                       ["clean"]
                       ["quality"]
                       ["test"]]}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [net.mikera/core.matrix "0.51.0"]])
