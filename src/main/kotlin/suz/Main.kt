package suz

import org.nd4j.linalg.api.ndarray.INDArray
import java.io.File

fun main(args: Array<String>) {

    val toriDataArray = loadFeatureArray(File("./data/1/"))
    val karasuDataArray = loadFeatureArray(File("./data/2/"))
//    var predict = NuralPrediction(toriDataArray, karasuDataArray)
//    predict.train(0.0001)
//    predict.save()
    rotateValidation(toriDataArray, karasuDataArray)
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