package suz

import com.google.common.collect.ImmutableList
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.io.File
import java.nio.charset.Charset

class LoadFeature {
    companion object {
        /**
         * 7x7の畳み込み画像を読み込む
         * 7x7の画像は特徴量が4方向用意されているので14x14の行列に拡張して処理を行う
         */
        @JvmStatic
        fun load(file: File): INDArray {
            var textString = file.readText(Charset.defaultCharset())
            val data = textString.split("\r\n|\n".toRegex())
            var featuresArray = mutableListOf<INDArray>()
            for (i: Int in 0..3) {
                var features = Nd4j.zeros(7, 7)
                for (j: Int in 0..6) {
                    for (k: Int in 0..6) {
                        features.put(j, k, Integer.parseInt(data[i * 7 * 7 + j * 7 + k]))
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
        fun loadFeatureArray(rootFolder: File): MutableList<INDArray> {
            var featureArray = mutableListOf<INDArray>()
            if (!rootFolder.isDirectory) return featureArray
            for (file in rootFolder.listFiles().filter { it.name.contains(".\\.txt".toRegex()) }.sorted()) {
                featureArray.add(load(file))
            }
            return featureArray
        }

        @JvmStatic
        fun loadLayer(file: File): MutableList<INDArray> {
            var textString = file.readText(Charset.defaultCharset())
            val data = textString.split("\n")
            var featuresArray = mutableListOf<INDArray>()
            for (i: Int in 0..3) {
                var features = Nd4j.zeros(7, 7)
                for (j: Int in 0..6) {
                    for (k: Int in 0..6) {
                        features.put(j, k, Integer.parseInt(data[i * 7 * 7 + j * 7 + k]))
                    }
                }
                featuresArray.add(features)
            }
            //14x14の行列を返す
            return featuresArray
        }
    }

}