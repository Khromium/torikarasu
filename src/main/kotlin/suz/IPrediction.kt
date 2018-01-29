package suz

import org.nd4j.linalg.api.ndarray.INDArray

interface IPrediction {

    /**
     * putargs
     */
    fun putArgs(args: Array<String>)

    /**
     * 学習
     */
    fun train(epsiron: Double = 0.0001)

    /**
     * 評価
     * @param xArray 評価用入力データ
     */
    fun varidation(xArray: INDArray): INDArray

    /**
     * 学習データ保存
     */
    fun save()

    /**
     * 学習データ読み込み
     */
    fun load()

}