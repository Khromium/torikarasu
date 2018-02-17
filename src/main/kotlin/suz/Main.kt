package suz

import org.apache.commons.math3.distribution.MultivariateNormalDistribution
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.io.File

fun main(args: Array<String>) {

    val toriDataArray = loadFeatureArray(File("./data/1/"))
    val karasuDataArray = loadFeatureArray(File("./data/2/"))
    var cov = arrayOf(doubleArrayOf(3.0, 1.0), doubleArrayOf(1.0, 3.0))
    var mnd1 = MultivariateNormalDistribution(doubleArrayOf(-5.0, -5.0), cov)
    var mnd2 = MultivariateNormalDistribution(doubleArrayOf(5.0, 5.0), cov)

    var dataArray1 = Nd4j.create(mnd1.sample(196))
    var dataArray2 = Nd4j.create(mnd2.sample(196))

//    var predict = NuralPrediction(toriDataArray, karasuDataArray)
//    predict.train(0.0001)
//    predict.save()
    var arrays1 = arrayListOf<INDArray>(dataArray1.getColumn(0))
    arrays1.add(dataArray1.getColumn(1))
    var arrays2 = arrayListOf<INDArray>(dataArray2.getColumn(0))
    arrays2.add(dataArray2.getColumn(1))
    var predict = LDA(arrays1, arrays2)
    predict.train()
    println(predict.result)
}

/**
 * ローテーション法による評価
 * @param targetNum 評価データとして用いる件数
 */
fun rotateValidation(toriDataArray: List<INDArray>, karasuDataArray: List<INDArray>, targetNum: Int = 20) {
    var count = 0
    if (toriDataArray.size != karasuDataArray.size) return //サイズ違いチェック
    if (toriDataArray.size % targetNum != 0 || karasuDataArray.size % targetNum != 0) return //割り切れるかチェック

    while (count + targetNum <= toriDataArray.size / 2) {
        var toriExtracted = mutableListOf<INDArray>()
        var toriEval = mutableListOf<INDArray>()
        var karasuExtracted = mutableListOf<INDArray>()
        var karasuEval = mutableListOf<INDArray>()
        toriDataArray.withIndex().filter { isRangeIn(count, targetNum, it.index) }.mapTo(toriEval) { it.value }
        karasuDataArray.withIndex().filter { isRangeIn(count, targetNum, it.index) }.mapTo(karasuEval) { it.value }
        toriDataArray.withIndex().filter { !isRangeIn(count, targetNum, it.index) }.mapTo(toriExtracted) { it.value }
        karasuDataArray.withIndex().filter { !isRangeIn(count, targetNum, it.index) }.mapTo(karasuExtracted) { it.value }
        var predict: IPrediction = LDA(toriExtracted, karasuExtracted) //認識データ指定
        predict.train(0.000005)

        var (xArray, indexArray) = createIndexedImageArray(toriEval, karasuEval) //評価用行列
        var result = predict.varidation(xArray)
        var toriCollect = 0
        var karasuCollect = 0
        for (index in 0..(targetNum * 2 - 1)) {//正答収集
            if (result.getDouble(0, index) > result.getDouble(1, index)
                    && indexArray.getDouble(0, index) > indexArray.getDouble(1, index)) toriCollect++  //TORI

            if (result.getDouble(0, index) < result.getDouble(1, index)
                    && indexArray.getDouble(0, index) < indexArray.getDouble(1, index)) karasuCollect++ //karasu
        }

        println("tori:${toriCollect}/${targetNum} ${toriCollect.toDouble() / targetNum * 100}%" +
                "\nkara:${karasuCollect}/${targetNum} ${karasuCollect.toDouble() / targetNum * 100}%" +
                "\n${toriCollect + karasuCollect}/${targetNum * 2} ${(toriCollect + karasuCollect).toDouble() / (targetNum * 2) * 100}%")
        count += targetNum //場所シフト
    }
}


fun isRangeIn(lower: Int, range: Int, index: Int): Boolean {
    if (index >= lower && index < (lower + range)) return true
    return false
}