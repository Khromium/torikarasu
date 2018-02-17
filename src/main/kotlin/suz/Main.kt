package suz

import org.apache.commons.math3.distribution.MultivariateNormalDistribution
import org.nd4j.linalg.factory.Nd4j

fun main(args: Array<String>) {

    var cov = arrayOf(doubleArrayOf(3.0, 1.0), doubleArrayOf(1.0, 3.0))
    var mnd1 = MultivariateNormalDistribution(doubleArrayOf(-5.0, -5.0), cov)
    var mnd2 = MultivariateNormalDistribution(doubleArrayOf(5.0, 5.0), cov)
    var dataArray1 = Nd4j.create(mnd1.sample(50))
    var dataArray2 = Nd4j.create(mnd2.sample(50))

    var predict = LDA(dataArray1, dataArray2)
    predict.train()

    println("w = [${predict.result!!.getDouble(0, 0)},${predict.result!!.getDouble(1, 0)}]")
}
