### poseidon
CUDA_VISIBLE_DEVICES=0 python train_SemanticKITTI.py --data_path /mnt/data/users/minesawa/semantickitti/growsp --sp_path /mnt/data/users/minesawa/semantickitti/growsp_sp --save_path /mnt/data/users/minesawa/semantickitti/tcuss_model_default_poseidon --workers 8 --temporal_workers 4 --cluster_workers 8

python train_SemanticKITTI.py --name newloss --data_path ~/dataset/semantickitti/growsp --sp_path ~/dataset/semantickitti/growsp_sp --patchwork_path ~/dataset/semantickitti/patchwork --save_path data/users/minesawa/semantickitti/newloss --workers 32 --temporal_workers 12 --cluster_workers 32 --batch_size 64 64
