package suz

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import java.io.File
import kotlin.math.absoluteValue

class Prediction(val toriDataArray: List<INDArray>, val karasuDataArray: List<INDArray>) {
    val HU = 42 //隠れ層
    val OU = 2 //出力：鳥か烏か 01 10
    val TORI = Nd4j.zeros(2, 1).put(0, 0, 1)
    val KARASU = Nd4j.zeros(2, 1).put(1, 0, 1)
    val LAMBDA = 0.0001
    val IU = Math.pow((ROWS_AND_COLUMNS * 2).toDouble(), 2.0).toInt()//画素サイズ 14*14

    var wWeightNDArray = Nd4j.randn(HU, IU).mul(0.1)
    var bBiasNDArray = Nd4j.randn(HU, 1).mul(0.1)
    var uWeightNDArray = Nd4j.randn(OU, HU).mul(0.1)
    var cBiasNDArray = Nd4j.randn(OU, 1).mul(0.1)


    /**
     * 学習部分
     * @param epsiron 閾値。前回との学習コストの差が epsiron 未満になったら停止する
     */
    fun train(epsiron: Double = 0.0001) {
        var lastCostValue = 99999.0
        var epoc = 1
        var images = mutableListOf<INDArray>()

        var indexArray = Nd4j.zeros(1, 1)

        var xArray = Nd4j.zeros(1, 1)

        toriDataArray.withIndex().forEach { (i, v) ->
            xArray = if (i != 0) Nd4j.hstack(xArray, v.reshape(IU, 1)) else v.reshape(IU, 1)
            indexArray = if (i != 0) Nd4j.hstack(indexArray, TORI) else TORI //Indexに鳥行列を追加
        }
        images.addAll(toriDataArray)

        karasuDataArray.withIndex().forEach { (i, v) ->
            xArray = Nd4j.hstack(xArray, v.reshape(IU, 1))
            indexArray = Nd4j.hstack(indexArray, KARASU) //Indexに烏行列を追加
        }
        images.addAll(karasuDataArray)

        while (true) {//エポック回し
            for ((index, input) in images.withIndex()) {

                val y = sigmoidNuron(input.reshape(IU, 1), wWeightNDArray, bBiasNDArray)
                val z = sigmoidNuron(y, uWeightNDArray, cBiasNDArray)
                val deltaOut = z.sub(indexArray.getColumn(index)).mul(z.mul(z.rsub(1)))
                val deltaHidden = deltaOut.transpose().mmul(uWeightNDArray).transpose().mul(y.mul(y.rsub(1)))

                val du = deltaOut.mmul(y.transpose())
                val dc = deltaOut.mul(-1)
                val dw = deltaHidden.mmul(input.reshape(IU, 1).transpose())
                val db = deltaHidden.mul(-1)

                uWeightNDArray = uWeightNDArray.sub(du.mul(LAMBDA))
                cBiasNDArray = cBiasNDArray.sub(dc.mul(LAMBDA))
                wWeightNDArray = wWeightNDArray.sub(dw.mul(LAMBDA))
                bBiasNDArray = bBiasNDArray.sub(db.mul(LAMBDA))

            }
            val ys = sigmoidNuron(xArray, wWeightNDArray, bBiasNDArray)
            val zs = sigmoidNuron(ys, uWeightNDArray, cBiasNDArray)
            val cost = costFunction(zs, indexArray)
            println("EPOC:${epoc}\tcost:${cost}")
            epoc++
            if ((lastCostValue - cost).absoluteValue <= epsiron) return //閾値以下になったら終了
            lastCostValue = cost
        }
    }

    /**
     * シグモイドニューロン
     * @param target 学習対象データ
     * @param weightArray 重み
     * @param biasVector バイアス
     */
    private fun sigmoidNuron(target: INDArray, weightArray: INDArray, biasVector: INDArray): INDArray {
        val p = weightArray.mmul(target).subColumnVector(biasVector).mul(-1)
        val denominator = Transforms.exp(p).add(1)
        return denominator.rdiv(1)
    }

    /**
     * コスト関数。正解データとの平均二乗誤差
     * @param target 評価対象のデータ
     * @param answer 正解データ
     * @return 計算結果
     */
    private fun costFunction(target: INDArray, answer: INDArray): Double {
        val c = target.sub(answer)
        return Transforms.pow(c, 2).sumNumber().toDouble()/(c.size(0)*c.size(1))
    }

    fun save() {
        DataSet(wWeightNDArray, Nd4j.zeros(1, 1)).save(File("./wWeight"))
        DataSet(bBiasNDArray, Nd4j.zeros(1, 1)).save(File("./bBias"))
        DataSet(uWeightNDArray, Nd4j.zeros(1, 1)).save((File("./uWeight")))
        DataSet(cBiasNDArray, Nd4j.zeros(1, 1)).save((File("./cBias")))
    }

    fun load(wWeightFile: File = File("./wWeight"),
             bBiasFile: File = File("./bBias"),
             uWeightFile: File = File("./uWeight"),
             cBiasFile: File = File("./cBias")) {
        var data = DataSet()
        data.load(wWeightFile)
        wWeightNDArray = data.featureMatrix
        data.load(bBiasFile)
        bBiasNDArray = data.featureMatrix
        data.load(uWeightFile)
        uWeightNDArray = data.featureMatrix
        data.load(cBiasFile)
        cBiasNDArray = data.featureMatrix
    }
}