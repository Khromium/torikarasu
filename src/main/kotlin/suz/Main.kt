package suz

import java.io.File

fun main(args: Array<String>) {

    val toriDataArray = loadFeatureArray(File("./data/1/"))
    val karasuDataArray = loadFeatureArray(File("./data/2/"))
    var predict = Prediction(toriDataArray, karasuDataArray)
    predict.train(epsiron = 1.0)
    predict.save()
}
