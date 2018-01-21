package suz

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import java.io.File
import kotlin.math.absoluteValue

val HU = 100 //隠れ層
val OU = 2 //出力：鳥か烏か 01 10
val TORI = Nd4j.zeros(2, 1).put(0, 0, 1)
val KARASU = Nd4j.zeros(2, 1).put(1, 0, 1)
val LAMBDA = 0.0001
val IU = Math.pow((ROWS_AND_COLUMNS * 2).toDouble(), 2.0).toInt()//画素サイズ 14*14


class Prediction(val toriDataArray: List<INDArray>, val karasuDataArray: List<INDArray>) : IPrediction {
    var wWeightNDArray = Nd4j.randn(HU, IU).mul(0.1) //入力層
    var bBiasNDArray = Nd4j.randn(HU, 1).mul(0.1) //入力層
    var vWeightNDArray = Nd4j.randn(HU, HU).mul(0.1) //中間層
    var dBiasNDArray = Nd4j.randn(HU, 1).mul(0.1) //中間層
    var uWeightNDArray = Nd4j.randn(OU, HU).mul(0.1) //出力層
    var cBiasNDArray = Nd4j.randn(OU, 1).mul(0.1) //出力層
    /**
     * 学習部分
     * @param epsiron 閾値。前回との学習コストの差が epsiron 未満になったら停止する
     */
    override fun train(epsiron: Double) {
        var lastCostValue = 99999.0
        var epoc = 1
        var images = mutableListOf<INDArray>()

        var (xArray, indexArray) = createIndexedImageArray()
        images.addAll(toriDataArray)
        images.addAll(karasuDataArray)

        while (true) {//エポック回し
            for ((index, input) in images.withIndex()) {

                val y = sigmoidNuron(input.reshape(IU, 1), wWeightNDArray, bBiasNDArray)
//                val x = sigmoidNuron(y, vWeightNDArray, dBiasNDArray)
//                val z = sigmoidNuron(x, uWeightNDArray, cBiasNDArray)
                val z = sigmoidNuron(y, uWeightNDArray, cBiasNDArray)
                val deltaOut = z.sub(indexArray.getColumn(index)).mul(z.mul(z.rsub(1)))
//                val deltaHidden2 = deltaOut.transpose().mmul(uWeightNDArray).transpose().mul(x.mul(x.rsub(1)))
//                val deltaHidden = deltaHidden2.transpose().mmul(vWeightNDArray).transpose().mul(y.mul(y.rsub(1)))
                val deltaHidden = deltaOut.transpose().mmul(uWeightNDArray).transpose().mul(y.mul(y.rsub(1)))

//                val du = deltaOut.mmul(x.transpose())
                val du = deltaOut.mmul(y.transpose())
                val dc = deltaOut.mul(-1)
//                val dv = deltaHidden2.mmul(y.transpose())
//                val dd = deltaHidden2.mul(-1)
                val dw = deltaHidden.mmul(input.reshape(IU, 1).transpose())
                val db = deltaHidden.mul(-1)

                uWeightNDArray = uWeightNDArray.sub(du.mul(LAMBDA))
                cBiasNDArray = cBiasNDArray.sub(dc.mul(LAMBDA))
//                vWeightNDArray = vWeightNDArray.sub(dv.mul(LAMBDA))
//                dBiasNDArray = dBiasNDArray.sub(dd.mul(LAMBDA))
                wWeightNDArray = wWeightNDArray.sub(dw.mul(LAMBDA))
                bBiasNDArray = bBiasNDArray.sub(db.mul(LAMBDA))

            }
            val ys = sigmoidNuron(xArray, wWeightNDArray, bBiasNDArray)
            val zs = sigmoidNuron(ys, uWeightNDArray, cBiasNDArray)
            val cost = costFunction(zs, indexArray)
            if (epoc % 50 == 1) println("EPOC:${epoc}\tcost:${cost}")

//            if (epoc>=500) {
            if ((lastCostValue - cost).absoluteValue <= epsiron) {
                println("EPOC:${epoc}\tcost:${cost}")
                return
            } //閾値以下になったら終了
            epoc++
            lastCostValue = cost
        }
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
    fun createIndexedImageArray(firstDataArray: List<INDArray> = toriDataArray, secondDataArray: List<INDArray> = karasuDataArray): Pair<INDArray, INDArray> {
        var indexArray = Nd4j.zeros(1, 1)
        var xArray = Nd4j.zeros(1, 1)
        firstDataArray.withIndex().forEach { (i, v) ->
            xArray = if (i != 0) Nd4j.hstack(xArray, v.reshape(IU, 1)) else v.reshape(IU, 1)
            indexArray = if (i != 0) Nd4j.hstack(indexArray, TORI) else TORI //Indexに鳥行列を追加
        }
        secondDataArray.withIndex().forEach { (i, v) ->
            xArray = Nd4j.hstack(xArray, v.reshape(IU, 1))
            indexArray = Nd4j.hstack(indexArray, KARASU) //Indexに烏行列を追加
        }
        return Pair(xArray, indexArray)
    }

    /**
     * 評価用関数
     * @param xArray 評価用データ
     */
    override fun varidation(xArray: INDArray): INDArray {
        val ys = sigmoidNuron(xArray, wWeightNDArray, bBiasNDArray)
        val zs = sigmoidNuron(ys, uWeightNDArray, cBiasNDArray)
        return zs
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
    fun costFunction(target: INDArray, answer: INDArray): Double {
        val c = target.sub(answer)
//        println(target)
        return Transforms.pow(c, 2).sumNumber().toDouble() / (c.size(0) * c.size(1))
    }


    /**
     * 学習データ保存
     */
    override fun save() {
        val wWeightFile = File("./wWeight")
        val bBiasFile = File("./bBias")
        val vWeightFile = File("./vWeight")
        val dBiasFile = File("./dBias")
        val uWeightFile = File("./uWeight")
        val cBiasFile = File("./cBias")
        DataSet(wWeightNDArray, Nd4j.zeros(1, 1)).save(wWeightFile)
        DataSet(bBiasNDArray, Nd4j.zeros(1, 1)).save(bBiasFile)
        DataSet(vWeightNDArray, Nd4j.zeros(1, 1)).save(vWeightFile)
        DataSet(dBiasNDArray, Nd4j.zeros(1, 1)).save(dBiasFile)
        DataSet(uWeightNDArray, Nd4j.zeros(1, 1)).save(uWeightFile)
        DataSet(cBiasNDArray, Nd4j.zeros(1, 1)).save(cBiasFile)
    }

    /**
     * 学習データ読み込み
     */
    override fun load() {
        val wWeightFile = File("./wWeight")
        val bBiasFile = File("./bBias")
        val vWeightFile = File("./vWeight")
        val dBiasFile = File("./dBias")
        val uWeightFile = File("./uWeight")
        val cBiasFile = File("./cBias")
        var data = DataSet()
        data.load(wWeightFile)
        wWeightNDArray = data.featureMatrix
        data.load(bBiasFile)
        bBiasNDArray = data.featureMatrix
        data.load(vWeightFile)
        vWeightNDArray = data.featureMatrix
        data.load(dBiasFile)
        dBiasNDArray = data.featureMatrix
        data.load(uWeightFile)
        uWeightNDArray = data.featureMatrix
        data.load(cBiasFile)
        cBiasNDArray = data.featureMatrix
    }
}