# 線形判別分析で文字認識を行う
※これは特別演習課題の一部です。
線形判別分析を使って特徴抽出済みの文字データを使用して文字認識を行います。
実装ではフィッシャーの線形判別と隠れ層1のNNでの判別のアルゴリズムを実装しています。

実装にあたっては全てKotlin+ND4Jで行っているので一応GPUで高速計算を行うことも可能です。(性能的には使わなくても十分ですが。)  
Kotlinの勉強しながら書いているので書き方が少し冗長な部分があります。

## 参考
[説明](https://qiita.com/Khromium/items/d31817154be0842326b0)
