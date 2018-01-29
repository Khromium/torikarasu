package suz

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.inverse.InvertMatrix
import java.io.File


class LDA(val toriDataArray: List<INDArray>, val karasuDataArray: List<INDArray>) : IPrediction {
    var result: INDArray? = null
    var start = 0
    var end = 0

    override fun putArgs(args: Array<String>) {
        start = args[0].toInt()
        end = args[1].toInt()
    }

    /**
     * 学習部分
     */
    override fun train(epsiron: Double) {
        val (toriTrain, karasuTrain) = createTKTrainArray()  //複数データを同時に扱いたかったので変形する
//        var mean1 = Nd4j.mean(toriTrain, 1) //次元0の平均計算
//        var mean2 = Nd4j.mean(karasuTrain, 1) //次元0の平均計算
        var mean1 = Nd4j.create(CalcMean(1, start, end)).transpose()
        var mean2 = Nd4j.create(CalcMean(2, start, end)).transpose()
//        var mean1 = mean(toriTrain.transpose()).transpose() //次元0の平均計算
//        var mean2 = mean(karasuTrain.transpose()).transpose() //次元0の平均計算
//        println(mean1)
//        println(mean2)
        // 総クラス内の共分散行列
        var sw = Nd4j.zeros(DATA_SIZE, DATA_SIZE)
        for (toridata in toriDataArray) {
            val shape = toridata.reshape(DATA_SIZE, 1)
            val sub = shape.sub(mean1)
            sw = sw.add(sub.mmul(sub.transpose()))
        }
        for (karasudata in karasuDataArray) {
            val shape = karasudata.reshape(DATA_SIZE, 1)
            val sub = shape.sub(mean2)
            sw = sw.add(sub.mmul(sub.transpose()))
        }
        var swInv = InvertMatrix.invert(sw, false)
        result = swInv.mmul(mean1.sub(mean2)) //傾きwを求めている
    }

    fun CalcMean(fileNum: Int, start: Int, end: Int): DoubleArray {
        var sum = DoubleArray(DATA_SIZE)
        for (num in 0 until 200) {
            if (isRangeIn(start, 20, num)) {
                var data = loadArray(File("./data/" + fileNum.toString() + "/" + String.format("%03d", fileNum) + ".txt"))
                for (index in 0 until DATA_SIZE) {
                    sum[index] += data[index]
                }
            }
        }
        for (index in 0 until DATA_SIZE) {
            sum[index] /= (end-start+1).toDouble()
        }
        return sum
    }

    fun mean(array: INDArray): INDArray {

        var retND = Nd4j.create(1, array.size(1))
        for (arrays in toriDataArray) {
            for (index in 0 until array.size(1)) {
                retND.put(0, index, array.getInt(0, index) + retND.getInt(0, index))
            }
        }
        for (arrays in karasuDataArray) {
            for (index in 0 until array.size(1)) {
                retND.put(0, index, array.getInt(0, index) + retND.getInt(0, index))
            }
        }
        return retND.div(toriDataArray.size + karasuDataArray.size)
//        for (index in 0 until array.size(1)) {
//            var col = array.getColumn(index)
//            var sum = 0.0
//            for (colindex in 0 until col.size(0)) {
//                sum += col.getInt(colindex, 0)
//            }
//            retND.put(0, index, sum / array.size(1))
//        }
//        return retND
    }

    fun createTKTrainArray(): Pair<INDArray, INDArray> {
        var toribase = toriDataArray[0].reshape(DATA_SIZE, 1)
        var karasubase = karasuDataArray[0].reshape(DATA_SIZE, 1)
        for (index in 1 until toriDataArray.size) {
            toribase = Nd4j.hstack(toribase, toriDataArray[index].reshape(DATA_SIZE, 1))
            karasubase = Nd4j.hstack(karasubase, karasuDataArray[index].reshape(DATA_SIZE, 1))
        }
        return Pair(toribase, karasubase)
    }


    /**
     * 評価用メソッド
     */
    override fun varidation(xArray: INDArray): INDArray {
        val (toriTrain, karasuTrain) = createTKTrainArray()
        val m1 = toriTrain.transpose().mmul(result).sumNumber().toDouble() / toriDataArray.size
        val m2 = karasuTrain.transpose().mmul(result).sumNumber().toDouble() / karasuDataArray.size
        var threshold = (m1 + m2) / 2 //判別のしきい値を求める
        println("threshold = $threshold")

        val yn = xArray.transpose().mmul(result) //結果計算
        var resultList1 = DoubleArray(yn.size(0))
        var resultList2 = DoubleArray(yn.size(0))
//        println("ynsize ${yn.size(0)}*${yn.size(1)}")

        for (yIndex in 0 until yn.size(0)) { //結果の行列を作ってる
            when {
                yn.getDouble(yIndex, 0) > threshold -> {
                    resultList1[yIndex] = 1.0
                    resultList2[yIndex] = 0.0
                }
                yn.getDouble(yIndex, 0) < threshold -> {
                    resultList1[yIndex] = 0.0
                    resultList2[yIndex] = 1.0
                }
                else -> {
                    resultList1[yIndex] = 0.0
                    resultList2[yIndex] = 0.0
                }
            }
        }
        return Nd4j.vstack(Nd4j.create(resultList1), Nd4j.create(resultList2))
    }

    override fun save() {
        val resultFile = File("./linearResult")
        DataSet(result, Nd4j.zeros(1, 1)).save(resultFile)
    }

    override fun load() {
        val resultFile = File("./linearResult")
        var data = DataSet()
        data.load(resultFile)
        result = data.featureMatrix
    }
}