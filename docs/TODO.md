# 現状
## 課題
- stc_lossとしてどのようなものがいいのかは模索中。異なる時間間で対応している点が多いsuperpoint同士の semantic primitives による分類の分布が近づくようにKLなどでLossを設定してもいいとか考えている。これからいいものを見つけていきたい。
- 道路について。現状では、init SPの時点で、歩道も道路と同じ1つの巨大なsuperpointとなってしまっている。init SPにて歩道はRANSACで含まれないようにしたい。→ 小さめにransacしても問題ないな多分 superpointが小さくなるだけだから。
- そもそもgrowsp_lossがこのタスクに適しているのかもわからない。growsp_lossが下がったからといって、val mIoU, val oAcc, val mAccが上がると
- 距離ごとの精度比較
- 移動物体と静止物体で比較
- VoteFlowは別のデータセットで学習されてしまっている。（SemanticKITTIでうまくいっているのか要確認）
- 反射率が考慮されていない。
    - init SP
    - backbone
- superpoint自体の移動量を入力 → 
<!-- TCUSSのベースラインの実装は終わっていて、テストした結果、ワールド座標系にて移動している物体（車や歩行者など）の精度は向上したが、静止している物体（建物や草木など）の精度は下がってしまった。そのため、静止している物体からのTARLのlossを計算しないようにして、TARL lossは動いている物体のみで計算して、誤差逆伝播したい。 -->

## TODO
growsp自体の精度向上 - epoch 30とかでk = 19でSuperpoint作って良くなるか。
- init spをましにする
- 再実行 → 全体にバグ or リトライにバグ 動くようにはなった。→ 精度低い。　persistanse? → 多分違う。GPU1~8でも変わらない。cluster前のにしてもダメ。→ dataloaderはpersistanceでも更新はされている。→ torchrun廃止してもダメ。マルチGPUだからではない。→batch sizeや！ or lr ß解決。
- trainsetとclustersetで別物になってしまっていた。
- STCは事前にgrowspでforwardしているんだから最適化できるはず。getitem時点でSP対応まで取っておきたい。ç
- 直ったら 
- Semantic Primitive 19でやってみる
- t_1とt_2で異なるdata augすれば、いい感じ？

train_SemanticKITTI.pyでは、マルチプロセスの設定をしているけど、これは確かpytorchのDataLoader内でcudaやgpuを使用したからだった気がする。今って、DataLoader内でgpuとか使うコードのまま？この設定は削除まだできない？確認して。default.yamlで実行している。


## 軽微な実装改善
- loss.backward一つにまとめる
- persistance対応
- evalのGPU使いきれていない感じ。CPUも8COREぐらいしか使っていない。
- clusterのバッチ対応とマルチGPU
- kmeansのマルチGPUの調整
- kmeans++


## 精度下がらない時
- lossも下がっていない場合
    - cluster_intervalごとに上昇する場合
        - 学習率と各epochごとのstep数を見直す。