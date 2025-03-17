## TCUSS: Temporal Consistent Unsupervised Semantic Segmentation of 3D Point Clouds

再現実験（GrowSP:itachi）
コマンドは適宜 CUDA_VISIBLE_DEVICES=xをつけること。
```
python train_SemanticKITTI.py --data_path data/users/minesawa/semantickitti/growsp --sp_path data/users/minesawa/semantickitti/growsp_sp --save_path data/users/minesawa/semantickitti/growsp_model_original
```

```
python train_SemanticKITTI.py --run_stage 1
```

```
python eval_SemanticKITTI.py --data_path data/users/minesawa/semantickitti/growsp --sp_path data/users/minesawa/semantickitti/growsp_sp --save_path data/users/minesawa/semantickitti/growsp_model
```


## submitまとめたコマンド
./test_SemanticKITTI.sh data/users/minesawa/semantickitti/growsp_model_original growsp.zip

### temp
#### invしなかった場合
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



### vis
CUDA_VISIBLE_DEVICES=3 WANDB_MODE=disabled python train_SemanticKITTI.py --data_path data/users/minesawa/semantickitti/growsp --sp_path data/users/minesawa/semantickitti/growsp_sp --save_path data/users/minesawa/semantickitti/tmp_model --growsp_start 80 --growsp_end 30 --run_stage 2 --vis

CUDA_VISIBLE_DEVICES=1 python train_SemanticKITTI_no_backprop.py --data_path data/users/minesawa/semantickitti/growsp --sp_path data/users/minesawa/semantickitti/growsp_sp --save_path data/users/minesawa/semantickitti/tcuss_model_no_backprop --workers 4 --temporal_workers 8 --cluster_workers 4

CUDA_VISIBLE_DEVICES=1 python train_SemanticKITTI_no_backprop.py --data_path data/users/minesawa/semantickitti/growsp --sp_path data/users/minesawa/semantickitti/growsp_sp --save_path data/users/minesawa/semantickitti/tcuss_model_no_backprop --workers 8 --temporal_workers 8 --cluster_workers 8 --batch_size 64 64行けるか？？？


### poseidon
CUDA_VISIBLE_DEVICES=0 python train_SemanticKITTI.py --data_path /mnt/data/users/minesawa/semantickitti/growsp --sp_path /mnt/data/users/minesawa/semantickitti/growsp_sp --save_path /mnt/data/users/minesawa/semantickitti/tcuss_model_default_poseidon --workers 8 --temporal_workers 4 --cluster_workers 8
CUDA_VISIBLE_DEVICES=1 python train_SemanticKITTI_no_backprop.py --data_path /mnt/data/users/minesawa/semantickitti/growsp --sp_path /mnt/data/users/minesawa/semantickitti/growsp_sp --save_path /mnt/data/users/minesawa/semantickitti/tcuss_model_no_backprop_poseidon --workers 8 --temporal_workers 4 --cluster_workers 8
CUDA_VISIBLE_DEVICES=2 python train_SemanticKITTI_growsp.py --data_path /mnt/data/users/minesawa/semantickitti/growsp --sp_path /mnt/data/users/minesawa/semantickitti/growsp_sp --save_path /mnt/data/users/minesawa/semantickitti/growsp_model_more_metric_poseidon --workers 8 --temporal_workers 4 --cluster_workers 8
CUDA_VISIBLE_DEVICES=3 python train_SemanticKITTI_hdb.py --data_path /mnt/data/users/minesawa/semantickitti/growsp --sp_path /mnt/data/users/minesawa/semantickitti/growsp_sp --save_path /mnt/data/users/minesawa/semantickitti/tcuss_model_hdb --workers 8 --temporal_workers 4 --cluster_workers 8 --hdb

CUDA_VISIBLE_DEVICES=0 python train_SemanticKITTI.py --data_path /mnt/data/users/minesawa/semantickitti/growsp --sp_path /mnt/data/users/minesawa/semantickitti/growsp_sp --save_path /mnt/data/users/minesawa/semantickitti/tcuss_model_default_poseidon --workers 8 --temporal_workers 4 --cluster_workers 8
CUDA_VISIBLE_DEVICES=0,1 python train_SemanticKITTI_onlyTARL.py --name onlyTARL_noScheduler --data_path /mnt/data/users/minesawa/semantickitti/growsp --sp_path /mnt/data/users/minesawa/semantickitti/growsp_sp --save_path /mnt/data/users/minesawa/semantickitti/onlyTARL_noScheduler --workers 16 --temporal_workers 6 --cluster_workers 16 --batch_size 16 16

python train_SemanticKITTI.py --name newloss --data_path ~/dataset/semantickitti/growsp --sp_path ~/dataset/semantickitti/growsp_sp --patchwork_path ~/dataset/semantickitti/patchwork --save_path data/users/minesawa/semantickitti/newloss --workers 32 --temporal_workers 12 --cluster_workers 32 --batch_size 64 64
### for debug
