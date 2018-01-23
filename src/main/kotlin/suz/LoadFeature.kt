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
 * 7x7の画像は特徴量が4方向用意されているので14x14の行列に拡張して処理を行う
 */
fun load(file: File): INDArray {
    var textString = file.readText(Charset.defaultCharset())
    val data = textString.split("\r\n|\n".toRegex())
    var featuresArray = mutableListOf<INDArray>()
    for (i: Int in 0..(FEATURE_NUM - 1)) {
        var features = Nd4j.zeros(ROWS_AND_COLUMNS, ROWS_AND_COLUMNS)
        for (j: Int in 0..(ROWS_AND_COLUMNS - 1)) {
            for (k: Int in 0..(ROWS_AND_COLUMNS - 1)) {
                features.put(j, k, Integer.parseInt(data[i * ROWS_AND_COLUMNS * ROWS_AND_COLUMNS + j * ROWS_AND_COLUMNS + k]))
            }
        }
        featuresArray.add(features)
    }
    //14x14の行列を返す
    return Nd4j.vstack(Nd4j.hstack(featuresArray[0], featuresArray[1]), Nd4j.hstack(featuresArray[2], featuresArray[3]))
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

fun loadLayer(file: File): MutableList<INDArray> {
    var textString = file.readText(Charset.defaultCharset())
    val data = textString.split("\n")
    var featuresArray = mutableListOf<INDArray>()
    for (i: Int in 0..(FEATURE_NUM - 1)) {
        var features = Nd4j.zeros(ROWS_AND_COLUMNS, ROWS_AND_COLUMNS)
        for (j: Int in 0..(ROWS_AND_COLUMNS - 1)) {
            for (k: Int in 0..(ROWS_AND_COLUMNS - 1)) {
                features.put(j, k, Integer.parseInt(data[i * ROWS_AND_COLUMNS * ROWS_AND_COLUMNS + j * ROWS_AND_COLUMNS + k]))
            }
        }
        featuresArray.add(features)
    }
    //14x14の行列を返す
    return featuresArray
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