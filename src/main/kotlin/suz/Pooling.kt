package suz

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex

class Pooling {
    /**
     * @param ndArray 入力行列
     * @param isMean true:平均値 false:最大値
     * @return プーリング済みの行列
     */
    fun poolingNDArray(ndArray: INDArray, isMean: Boolean): INDArray {
        val H = 2 //region size is 2x2. shift size is also 2
        if (ndArray.size(0) % H != 0 || ndArray.size(1) % H != 0)
            throw IllegalArgumentException("Can't pooling")
        var retNDArray = Nd4j.zeros(ndArray.size(0) / H, ndArray.size(1) / H)//result ndArray
        var rowPoint = H
        var columnPoint = H
        while (rowPoint <= ndArray.size(1)) {//行の終わりまでみる
            while (columnPoint <= ndArray.size(0)) {//列の終わりまでみる
                val region = ndArray.get(NDArrayIndex.interval(rowPoint - H, rowPoint), NDArrayIndex.interval(columnPoint - H, columnPoint))//領域切り出し
                if (isMean) {
                    retNDArray.put(rowPoint / H - 1, columnPoint / H - 1, region.meanNumber())//平均値
                } else {
                    retNDArray.put(rowPoint / H - 1, columnPoint / H - 1, region.maxNumber())//最大値
                }
                columnPoint += H
            }
            columnPoint = H //reset
            rowPoint += H //shift
        }
        return retNDArray
    }

    fun backward(){

    }
}