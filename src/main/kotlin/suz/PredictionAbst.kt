package suz

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import java.io.File

abstract class PredictionAbst {
    var wWeightNDArray = Nd4j.randn(HU, IU).mul(0.1) //入力層
    var bBiasNDArray = Nd4j.randn(HU, 1).mul(0.1) //入力層
    var vWeightNDArray = Nd4j.randn(HU, HU).mul(0.1) //中間層
    var dBiasNDArray = Nd4j.randn(HU, 1).mul(0.1) //中間層
    var uWeightNDArray = Nd4j.randn(OU, HU).mul(0.1) //出力層
    var cBiasNDArray = Nd4j.randn(OU, 1).mul(0.1) //出力層
    abstract fun train(epsiron: Double = 0.0001)
    abstract fun varidation(xArray: INDArray): INDArray

    /**
     * 学習データ保存
     */
    fun save(wWeightFile: File = File("./wWeight"),
             bBiasFile: File = File("./bBias"),
             vWeightFile: File = File("./vWeight"),
             dBiasFile: File = File("./dBias"),
             uWeightFile: File = File("./uWeight"),
             cBiasFile: File = File("./cBias")) {
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
    fun load(wWeightFile: File = File("./wWeight"),
             bBiasFile: File = File("./bBias"),
             vWeightFile: File = File("./vWeight"),
             dBiasFile: File = File("./dBias"),
             uWeightFile: File = File("./uWeight"),
             cBiasFile: File = File("./cBias")) {
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