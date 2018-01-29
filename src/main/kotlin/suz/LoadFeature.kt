package suz

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.io.File
import java.nio.charset.Charset

const val ROWS_AND_COLUMNS = 7 //画素数
const val FEATURE_NUM = 4//特徴量の数
val DATA_SIZE = Math.pow((ROWS_AND_COLUMNS * 2).toDouble(), 2.0).toInt()//画素サイズ 14*14

/**
 * 7x7の畳み込み画像を読み込む
 */
fun load(file: File): INDArray {
    var textString = file.readText(Charset.defaultCharset())
    var data = textString.split("\r\n|\n".toRegex())
    var dataArray = DoubleArray(DATA_SIZE)
    data.withIndex().forEach { if (it.index < DATA_SIZE) dataArray[it.index] = it.value.toDouble() }
    return Nd4j.create(dataArray)
}

fun loadArray(file: File): DoubleArray {
    var textString = file.readText(Charset.defaultCharset())
    var data = textString.split("\r\n|\n".toRegex())
    var dataArray = DoubleArray(DATA_SIZE)
    data.withIndex().forEach { if (it.index < DATA_SIZE) dataArray[it.index] = it.value.toDouble() }
    return dataArray
}

/**
 * 一括ファイル読み込み
 */
fun loadFeatureArray(rootFolder: File): List<INDArray> {
    var featureArray = mutableListOf<INDArray>()
    if (!rootFolder.isDirectory) return featureArray
    rootFolder.listFiles().filter { it.name.contains(".\\.txt".toRegex()) }.sorted().mapTo(featureArray) { load(it) }
    return featureArray
}


/**
 * データ集合を一枚一枚縦に並べたデータの行列に変更します。
 * 鳥,烏の順に作られます。
 *
 * それに応じて鳥か烏かの正解データも返します
 * @param firstDataArray 1番目(鳥)データ
 * @param secondDataArray 2番目(烏)データ
 * @return 行列データ,正解データ
 */
fun createIndexedImageArray(firstDataArray: List<INDArray>, secondDataArray: List<INDArray>): Pair<INDArray, INDArray> {
    var indexArray = Nd4j.zeros(1, 1)
    var xArray = Nd4j.zeros(1, 1)
    firstDataArray.withIndex().forEach { (i, v) ->
        xArray = if (i != 0) Nd4j.hstack(xArray, v.reshape(DATA_SIZE, 1)) else v.reshape(DATA_SIZE, 1)
        indexArray = if (i != 0) Nd4j.hstack(indexArray, TORI) else TORI //Indexに鳥行列を追加
    }
    secondDataArray.withIndex().forEach { (i, v) ->
        xArray = Nd4j.hstack(xArray, v.reshape(DATA_SIZE, 1))
        indexArray = Nd4j.hstack(indexArray, KARASU) //Indexに烏行列を追加
    }
    return Pair(xArray, indexArray)
}