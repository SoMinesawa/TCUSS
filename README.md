## TCUSS: Temporal Consistent Unsupervised Semantic Segmentation of 3D Point Clouds

再現実験（GrowSP:itachi）
コマンドは適宜 CUDA_VISIBLE_DEVICES=xをつけること。
```
python train_SemanticKITTI.py --data_path /mnt/urashima/users/minesawa/semantickitti/growsp --sp_path /mnt/urashima/users/minesawa/semantickitti/growsp_sp --save_path /mnt/urashima/users/minesawa/semantickitti/growsp_model_original
```

```
python train_SemanticKITTI.py --run_stage 1
```

```
python eval_SemanticKITTI.py --data_path /mnt/urashima/users/minesawa/semantickitti/growsp --sp_path /mnt/urashima/users/minesawa/semantickitti/growsp_sp --save_path /mnt/urashima/users/minesawa/semantickitti/growsp_model
```


## How to submit
```
python test_SemanticKITTI.py --data_path /mnt/urashima/users/minesawa/semantickitti/growsp --sp_path /mnt/urashima/users/minesawa/semantickitti/growsp_sp --save_path /mnt/urashima/users/minesawa/semantickitti/growsp_model_select3000
--debug
```

### semantic-kitti-api
以下、~/repos/semantic-kitti-api上で実行、不安だったら、/mnt/urashima/users/minesawa/semantickitti/a.ipynbでデータの形状確認する
```
python evaluate_semantics.py -d /mnt/urashima/dataset/semantickitti/dataset/ -p /mnt/urashima/users/minesawa/semantickitti/growsp_model_select3000/pred_result/
```

```
/mnt/urashima/users/minesawa/semantickitti/growsp_model_select3000/pred_result$ zip -r first_submission.zip .
```

```
python validate_submission.py /mnt/urashima/users/minesawa/semantickitti/growsp_model_select3000/pred_result/first_submission.zip /mnt/urashima/dataset/semantickitti/dataset/
```

### submitまとめたコマンド
./test_SemanticKITTI.sh /mnt/urashima/users/minesawa/semantickitti/growsp_model_original growsp.zip
```
MODEL_DIR="/mnt/urashima/users/minesawa/semantickitti/growsp_model_original" && SUBMISSION_NAME="growsp.zip" && DATASET_PATH="/mnt/urashima/dataset/semantickitti/dataset/" && \
python test_SemanticKITTI.py --data_path /mnt/urashima/users/minesawa/semantickitti/growsp --sp_path /mnt/urashima/users/minesawa/semantickitti/growsp_sp --save_path "$MODEL_DIR" --debug && \
cd ~/repos/semantic-kitti-api && \
python evaluate_semantics.py -d "$DATASET_PATH" -p "$MODEL_DIR/pred_result/" && \
cd "$MODEL_DIR/pred_result" && \
zip -r "$SUBMISSION_NAME" . && \
python ~/repos/semantic-kitti-api/validate_submission.py "$MODEL_DIR/pred_result/$SUBMISSION_NAME" "$DATASET_PATH" && \
cd ~/repos/TCUSS && \
$ scp -P 2222 -i ~/.ssh/id_ishikawa /mnt/urashima/users/minesawa/semantickitti/growsp_model_original/pred_result/growsp.zip somin@localhost:"C:\Users\somin\Downloads"
```

### temp
#### invしなかった場合
Acc avg 0.002
IoU avg 0.000
IoU class 1 [car] = 0.002
IoU class 2 [bicycle] = 0.000
IoU class 3 [motorcycle] = 0.003
IoU class 4 [truck] = 0.000
IoU class 5 [other-vehicle] = 0.003
IoU class 6 [person] = 0.000
IoU class 7 [bicyclist] = 0.000
IoU class 8 [motorcyclist] = 0.000
IoU class 9 [road] = 0.000
IoU class 10 [parking] = 0.000
IoU class 11 [sidewalk] = 0.000
IoU class 12 [other-ground] = 0.000
IoU class 13 [building] = 0.000
IoU class 14 [fence] = 0.000
IoU class 15 [vegetation] = 0.000
IoU class 16 [trunk] = 0.000
IoU class 17 [terrain] = 0.000
IoU class 18 [pole] = 0.000
IoU class 19 [traffic-sign] = 0.000
### invした場合
Acc avg 0.011
IoU avg 0.003
IoU class 1 [car] = 0.001
IoU class 2 [bicycle] = 0.000
IoU class 3 [motorcycle] = 0.000
IoU class 4 [truck] = 0.003
IoU class 5 [other-vehicle] = 0.001
IoU class 6 [person] = 0.001
IoU class 7 [bicyclist] = 0.001
IoU class 8 [motorcyclist] = 0.000
IoU class 9 [road] = 0.000
IoU class 10 [parking] = 0.019
IoU class 11 [sidewalk] = 0.000
IoU class 12 [other-ground] = 0.000
IoU class 13 [building] = 0.001
IoU class 14 [fence] = 0.001
IoU class 15 [vegetation] = 0.027
IoU class 16 [trunk] = 0.004
IoU class 17 [terrain] = 0.000
IoU class 18 [pole] = 0.000
IoU class 19 [traffic-sign] = 0.000

val + test → こっちで提出してみるか
[[ 0  7]
 [ 1 16]
 [ 2 13]
 [ 3  9]
 [ 4 12]
 [ 5  3]
 [ 6 14]
 [ 7  5]
 [ 8 18]
 [ 9  1]
 [10 10]
 [11  8]
 [12  4]
 [13  0]
 [14  2]
 [15 15]
 [16  6]
 [17 17]
 [18 11]]
Acc avg 0.441
IoU avg 0.115
mean IoU,mean accuracy,car,bicycle,motorcycle,truck,other-vehicle,person,bicyclist,motorcyclist,road,parking,sidewalk,other-ground,building,fence,vegetation,trunk,terrain,pole,traffic-sign
12.7,48.4,84.4,0.0,0.0,0.0,0.8,0.1,0.0,0.0,34.8,0.0,4.3,2.2,46.8,2.5,38.7,2.3,23.2,0.1,0.4

train + val + test
[[ 0  7]
 [ 1 13]
 [ 2  9]
 [ 3  5]
 [ 4  0]
 [ 5  3]
 [ 6 11]
 [ 7 16]
 [ 8 18]
 [ 9  1]
 [10 10]
 [11 14]
 [12  4]
 [13  8]
 [14  2]
 [15 15]
 [16  6]
 [17 17]
 [18 12]]
Acc avg 0.441
IoU avg 0.114

一回GrowSPで精度出るのか提出してみるか


### vis
CUDA_VISIBLE_DEVICES=3 WANDB_MODE=disabled python train_SemanticKITTI.py --data_path /mnt/urashima/users/minesawa/semantickitti/growsp --sp_path /mnt/urashima/users/minesawa/semantickitti/growsp_sp --save_path /mnt/urashima/users/minesawa/semantickitti/tmp_model --growsp_start 80 --growsp_end 30 --run_stage 2 --vis

CUDA_VISIBLE_DEVICES=1 python train_SemanticKITTI_no_backprop.py --data_path /mnt/urashima/users/minesawa/semantickitti/growsp --sp_path /mnt/urashima/users/minesawa/semantickitti/growsp_sp --save_path /mnt/urashima/users/minesawa/semantickitti/tcuss_model_no_backprop --workers 4 --temporal_workers 8 --cluster_workers 4

CUDA_VISIBLE_DEVICES=1 python train_SemanticKITTI_no_backprop.py --data_path /mnt/urashima/users/minesawa/semantickitti/growsp --sp_path /mnt/urashima/users/minesawa/semantickitti/growsp_sp --save_path /mnt/urashima/users/minesawa/semantickitti/tcuss_model_no_backprop --workers 8 --temporal_workers 8 --cluster_workers 8 --batch_size 64 64行けるか？？？


### poseidon
CUDA_VISIBLE_DEVICES=0 python train_SemanticKITTI.py --data_path /mnt/data/users/minesawa/semantickitti/growsp --sp_path /mnt/data/users/minesawa/semantickitti/growsp_sp --save_path /mnt/data/users/minesawa/semantickitti/tcuss_model_default_poseidon --workers 8 --temporal_workers 4 --cluster_workers 8
CUDA_VISIBLE_DEVICES=1 python train_SemanticKITTI_no_backprop.py --data_path /mnt/data/users/minesawa/semantickitti/growsp --sp_path /mnt/data/users/minesawa/semantickitti/growsp_sp --save_path /mnt/data/users/minesawa/semantickitti/tcuss_model_no_backprop_poseidon --workers 8 --temporal_workers 4 --cluster_workers 8
CUDA_VISIBLE_DEVICES=2 python train_SemanticKITTI_growsp.py --data_path /mnt/data/users/minesawa/semantickitti/growsp --sp_path /mnt/data/users/minesawa/semantickitti/growsp_sp --save_path /mnt/data/users/minesawa/semantickitti/growsp_model_more_metric_poseidon --workers 8 --temporal_workers 4 --cluster_workers 8
CUDA_VISIBLE_DEVICES=3 python train_SemanticKITTI_hdb.py --data_path /mnt/data/users/minesawa/semantickitti/growsp --sp_path /mnt/data/users/minesawa/semantickitti/growsp_sp --save_path /mnt/data/users/minesawa/semantickitti/tcuss_model_hdb --workers 8 --temporal_workers 4 --cluster_workers 8 --hdb

CUDA_VISIBLE_DEVICES=0 python train_SemanticKITTI.py --data_path /mnt/data/users/minesawa/semantickitti/growsp --sp_path /mnt/data/users/minesawa/semantickitti/growsp_sp --save_path /mnt/data/users/minesawa/semantickitti/tcuss_model_default_poseidon --workers 8 --temporal_workers 4 --cluster_workers 8
CUDA_VISIBLE_DEVICES=0,1 python train_SemanticKITTI_onlyTARL.py --name onlyTARL_noScheduler --data_path /mnt/data/users/minesawa/semantickitti/growsp --sp_path /mnt/data/users/minesawa/semantickitti/growsp_sp --save_path /mnt/data/users/minesawa/semantickitti/onlyTARL_noScheduler --workers 16 --temporal_workers 6 --cluster_workers 16 --batch_size 16 16
### for debug
