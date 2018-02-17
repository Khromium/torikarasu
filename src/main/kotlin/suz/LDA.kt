package suz

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.inverse.InvertMatrix
import java.io.File


class LDA(val classData1: INDArray, val classData2: INDArray) {
    var result: INDArray? = null
    val DATA_SIZE = 2

    /**
     * 学習部分
     */
    fun train(): INDArray? {
        var mean1 = Nd4j.mean(classData1, 0).reshape(classData1.size(1), 1) //次元0の平均計算
        var mean2 = Nd4j.mean(classData2, 0).reshape(classData2.size(1), 1) //次元0の平均計算
        //総クラス内の共分散行列
        var sw = Nd4j.zeros(DATA_SIZE, DATA_SIZE)
        for (index in 0 until classData1.size(0)) {
            val shape = classData1.transpose().getColumn(index).reshape(DATA_SIZE, 1)
            val sub = shape.sub(mean1)
            sw = sw.add(sub.mmul(sub.transpose()))
        }
        for (index in 0 until classData2.size(0)) {
            val shape = classData2.transpose().getColumn(index).reshape(DATA_SIZE, 1)
            val sub = shape.sub(mean2)
            sw = sw.add(sub.mmul(sub.transpose()))
        }
        var swInv = InvertMatrix.invert(sw, false)
        result = swInv.mmul(mean1.sub(mean2)) //傾きwを求めている
        return result
    }


    fun save() {
        val resultFile = File("./linearResult")
        DataSet(result, Nd4j.zeros(1, 1)).save(resultFile)
    }

    fun load() {
        val resultFile = File("./linearResult")
        var data = DataSet()
        data.load(resultFile)
        result = data.featureMatrix
    }
}