package suz

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import java.io.File

fun main(args: Array<String>) {
    val HU = 196
    val OU = 2 //鳥か烏か 01 10
    val TORI = Nd4j.zeros(2, 1).put(0, 0, 1)
    val KARASU = Nd4j.zeros(2, 1).put(1, 0, 1)
    val LAMBDA = 0.0001

    var images = LoadFeature.loadFeatureArray(File("./data/1/"))
    val IU = images[0].size(0) * images[0].size(1)
    var indexArray = Nd4j.zeros(1, 1)

    var xArray = Nd4j.zeros(1, 1)
    images.withIndex().forEach { (i, v) ->
        xArray = if (i != 0) Nd4j.hstack(xArray, v.reshape(IU, 1)) else v.reshape(IU, 1)
        indexArray = if (i != 0) Nd4j.hstack(indexArray, TORI) else TORI //Indexに鳥行列を追加
    }
    val images2 = LoadFeature.loadFeatureArray(File("./data/2/"))
    images2.withIndex().forEach { (i, v) ->
        xArray = Nd4j.hstack(xArray, v.reshape(IU, 1))
        indexArray = Nd4j.hstack(indexArray, KARASU) //Indexに烏行列を追加
    }
    images.addAll(images2)

    var wWeightNDArray = Nd4j.randn(HU, IU).mul(0.1)
    var bBiasNDArray = Nd4j.randn(HU, 1).mul(0.1)
    var uWeightNDArray = Nd4j.randn(OU, HU).mul(0.1)
    var cBiasArray = Nd4j.randn(OU, 1).mul(0.1)

    while (true) {//エポック回し
        for ((index, input) in images.withIndex()) {
            val y = sigmoidNuron(input.reshape(IU, 1), wWeightNDArray, bBiasNDArray)
            val z = sigmoidNuron(y, uWeightNDArray, cBiasArray)
            val deltaOut = z.sub(indexArray.getColumn(index)).mul(z.mul(z.rsub(1)))
            val deltaHidden = deltaOut.transpose().mmul(uWeightNDArray).transpose().mul(y.mul(y.rsub(1)))

            val du = deltaOut.mmul(y.transpose())
            val dc = deltaOut.mul(-1)
            val dw = deltaHidden.mmul(input.reshape(IU, 1).transpose())
            val db = deltaHidden.mul(-1)

            uWeightNDArray = uWeightNDArray.sub(du.mul(LAMBDA))
            cBiasArray = cBiasArray.sub(dc.mul(LAMBDA))
            wWeightNDArray = wWeightNDArray.sub(dw.mul(LAMBDA))
            bBiasNDArray = bBiasNDArray.sub(db.mul(LAMBDA))

        }
//        for (input in images) {
//            val ys = sigmoidNuron(input.reshape(IU, 1), wWeightNDArray, bBiasNDArray)
//            val zs = sigmoidNuron(ys, uWeightNDArray, cBiasArray)
//            println(zs)
//        }
        val ys = sigmoidNuron(xArray, wWeightNDArray, bBiasNDArray)
        val zs = sigmoidNuron(ys, uWeightNDArray, cBiasArray)
//        println(ys)
//        println(zs)
        val cost = costFunction(zs, indexArray).sumNumber().toDouble()
//        for (img in images) {
//            val ys1 = sigmoidNuron(img.reshape(IU, 1), wWeightNDArray, bBiasNDArray)
//            val zs1 = sigmoidNuron(ys1, uWeightNDArray, cBiasArray)
//            println("test$zs1")
//        }

        println(cost)

    }
}

fun sigmoidNuron(target: INDArray, weightArray: INDArray, biasVector: INDArray): INDArray {
    val p = weightArray.mmul(target).subColumnVector(biasVector).mul(-1)
    val denominator = Transforms.exp(p).add(1)
    return denominator.rdiv(1)
}

fun costFunction(target: INDArray, answer: INDArray): INDArray {
    println(target)
//    println(answer)
    val c = target.sub(answer)
//    println(c)
    return Nd4j.sum(Transforms.pow(c, 2))
}