package suz

import org.nd4j.linalg.api.ndarray.INDArray

interface IPrediction {

    /**
     * 学習
     */
    fun train(epsiron: Double = 0.0001)

    /**
     * 評価
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