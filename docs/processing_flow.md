# TCUSS 処理フロー

このドキュメントでは、TCUSSの主要な処理フロー（init SP, backbone, Superpoint Constructor, SPC）における入出力を図示します。

## 全体フロー概要

```mermaid
flowchart TB
    subgraph Preprocessing["前処理 (initialSP_prepare_SemanticKITTI.py)"]
        RAW_PLY["点群 PLY<br/>(x, y, z, class)"]
        COORDS_ONLY["座標のみ抽出<br/>(x, y, z)"]
        RANSAC["RANSAC<br/>地面平面推定"]
        DBSCAN["DBSCAN<br/>非地面点クラスタリング"]
        INIT_SP["init SP ラベル<br/>(N,) int"]
        
        RAW_PLY --> COORDS_ONLY
        COORDS_ONLY --> RANSAC
        COORDS_ONLY --> DBSCAN
        RANSAC --> INIT_SP
        DBSCAN --> INIT_SP
    end

    subgraph Training["学習ループ (train_SemanticKITTI.py)"]
        subgraph Clustering["クラスタリングフェーズ (cluster_interval毎)"]
            VOXEL_COORDS["Voxel座標<br/>(N, 3)"]
            BACKBONE_C["Backbone<br/>(Res16FPN18)"]
            POINT_FEATS_C["点特徴量<br/>(N, 128)"]
            SP_CONSTRUCTOR["Superpoint<br/>Constructor"]
            SP_FEATS["SP特徴量<br/>(M, 139)"]
            SPC["SPC<br/>(k-means)"]
            PRIMITIVES["Primitiveラベル<br/>+ Classifier"]
            
            VOXEL_COORDS --> BACKBONE_C
            BACKBONE_C --> POINT_FEATS_C
            POINT_FEATS_C --> SP_CONSTRUCTOR
            SP_CONSTRUCTOR --> SP_FEATS
            SP_FEATS --> SPC
            SPC --> PRIMITIVES
        end
        
        subgraph TrainStep["学習ステップ (各バッチ)"]
            VOXEL_COORDS_T["Voxel座標<br/>(N, 3)"]
            BACKBONE_T["Backbone<br/>(Res16FPN18)"]
            POINT_FEATS_T["点特徴量<br/>(N, 128)"]
            CLASSIFIER["Classifier<br/>(固定重み)"]
            LOSS["CrossEntropy Loss"]
            
            VOXEL_COORDS_T --> BACKBONE_T
            BACKBONE_T --> POINT_FEATS_T
            POINT_FEATS_T --> CLASSIFIER
            CLASSIFIER --> LOSS
        end
        
        PRIMITIVES -.->|"疑似ラベル"| LOSS
    end

    INIT_SP -.->|"読み込み"| SP_CONSTRUCTOR
```

---

## 各処理の詳細

### 1. init SP の作成

**ファイル**: `data_prepare/initialSP_prepare_SemanticKITTI.py`

```mermaid
flowchart LR
    subgraph Input["入力"]
        COORDS["座標 (x, y, z)<br/>shape: (N, 3)<br/>float32"]
    end
    
    subgraph Process["処理"]
        RANSAC["RANSAC<br/>distance_threshold=0.1<br/>ransac_n=3<br/>num_iterations=1000"]
        DBSCAN["DBSCAN<br/>eps=0.2<br/>min_points=1"]
    end
    
    subgraph Output["出力"]
        SP_LABELS["SP ラベル<br/>shape: (N,)<br/>int"]
    end
    
    COORDS --> RANSAC
    COORDS --> DBSCAN
    RANSAC -->|"地面点インデックス"| SP_LABELS
    DBSCAN -->|"非地面点クラスタID"| SP_LABELS

    style Input fill:#e1f5fe
    style Output fill:#c8e6c9
```

**ポイント**:
- ✅ 使用: 座標 (x, y, z) のみ
- ❌ 未使用: 強度 (remission), クラスラベル (GT)
- クラスラベルは品質評価（mIoU計算）のみに使用

---

### 2. Backbone (Res16FPN18)

**ファイル**: `models/fpn.py`, `lib/trainer.py`

```mermaid
flowchart LR
    subgraph Input["入力"]
        VOXEL["Voxel座標<br/>shape: (N, 3)<br/>× voxel_size (0.15m)"]
    end
    
    subgraph Model["Res16FPN18"]
        CONV0["Conv0 (k=5)"]
        BLOCK1["Block1 (32ch)"]
        BLOCK2["Block2 (64ch)"]
        BLOCK3["Block3 (128ch)"]
        BLOCK4["Block4 (256ch)"]
        FPN["FPN Fusion<br/>(4スケール補間加算)"]
        
        CONV0 --> BLOCK1 --> BLOCK2 --> BLOCK3 --> BLOCK4 --> FPN
    end
    
    subgraph Output["出力"]
        FEATS["点特徴量<br/>shape: (N, 128)<br/>float32"]
    end
    
    VOXEL --> CONV0
    FPN --> FEATS

    style Input fill:#e1f5fe
    style Output fill:#c8e6c9
```

**ポイント**:
- ✅ 使用: Voxel座標 (x, y, z) のみ (`input_dim=3`)
- ❌ 未使用: 強度 (remission), 法線, 色
- MinkowskiEngine による Sparse 3D Convolution

---

### 3. Superpoint Constructor (GrowSP)

**ファイル**: `lib/utils.py` (`get_kittisp_feature` 関数)

```mermaid
flowchart TB
    subgraph Input["入力"]
        POINT_FEATS["点特徴量<br/>(backbone出力)<br/>shape: (N, 128)"]
        REMISSION["強度 (remission)<br/>shape: (N, 1)"]
        NORMALS["法線<br/>shape: (N, 3)"]
        INIT_SP["init SPラベル<br/>shape: (N,)"]
    end
    
    subgraph Step1["Step 1: init SP 特徴量計算"]
        AGG1["init SP内で平均"]
        REGION_FEATS["init SP特徴量<br/>(M₀, 128)"]
        
        POINT_FEATS --> AGG1
        INIT_SP --> AGG1
        AGG1 --> REGION_FEATS
    end
    
    subgraph Step2["Step 2: k-means でSP統合"]
        KMEANS1["k-means<br/>n_clusters=current_growsp"]
        SP_IDX["SP統合ラベル<br/>(M₀,) → M個に統合"]
        
        REGION_FEATS --> KMEANS1
        KMEANS1 --> SP_IDX
    end
    
    subgraph Step3["Step 3: 統合SP特徴量計算"]
        AGG2["統合SP内で平均"]
        FINAL_FEATS["SP点特徴量<br/>(M, 128)"]
        FINAL_REM["SP強度<br/>(M, 1)"]
        PFH["PFH (法線ヒストグラム)<br/>(M, 10)"]
        
        POINT_FEATS --> AGG2
        SP_IDX --> AGG2
        REMISSION --> AGG2
        NORMALS --> AGG2
        
        AGG2 --> FINAL_FEATS
        AGG2 --> FINAL_REM
        AGG2 --> PFH
    end
    
    subgraph Output["出力"]
        CONCAT["特徴量連結"]
        SP_FEATS["SP特徴量<br/>shape: (M, 139)<br/>= 128 + 1×c_rgb + 10×c_shape"]
        
        FINAL_FEATS --> CONCAT
        FINAL_REM -->|"×5.0 (c_rgb)"| CONCAT
        PFH -->|"×5.0 (c_shape)"| CONCAT
        CONCAT --> SP_FEATS
    end

    style Input fill:#e1f5fe
    style Output fill:#c8e6c9
```

**ポイント**:
- ✅ 使用:
  - 点特徴量 (backbone出力): 128次元
  - 強度 (remission): 1次元 × c_rgb (5.0)
  - 法線 → PFH: 10次元 × c_shape (5.0)
  - init SPラベル: 点をSPにグループ化
- GrowSP: エポック進行に伴い `current_growsp` を 80 → 30 に減少

---

### 4. Semantic Primitive Clustering (SPC)

**ファイル**: `lib/trainer.py` (`cluster` 関数), `lib/utils.py`

```mermaid
flowchart LR
    subgraph Input["入力"]
        SP_FEATS["全シーンのSP特徴量<br/>shape: (総SP数, 139)<br/>= backbone(128) + remission(1) + PFH(10)"]
    end
    
    subgraph Process["k-means クラスタリング"]
        KMEANS["k-means<br/>n_clusters=500<br/>(primitive_num)"]
    end
    
    subgraph Output["出力"]
        PRIM_LABELS["Primitiveラベル<br/>shape: (総SP数,)<br/>値: 0~499"]
        PRIM_CENTERS["Primitiveセンター<br/>shape: (500, 128)<br/>※幾何特徴は除外"]
        CLASSIFIER["Classifier<br/>(固定重み分類器)"]
    end
    
    SP_FEATS --> KMEANS
    KMEANS --> PRIM_LABELS
    KMEANS --> PRIM_CENTERS
    PRIM_CENTERS --> CLASSIFIER

    style Input fill:#e1f5fe
    style Output fill:#c8e6c9
```

**ポイント**:
- ✅ 使用: SP特徴量 (139次元) 全体でk-means
- 出力の分類器重みは backbone特徴 (128次元) のみ使用
- `select_num=1500` シーンからSP特徴を収集

---

## 情報フローサマリー

```mermaid
flowchart TB
    subgraph Data["データ属性"]
        XYZ["座標<br/>(x, y, z)"]
        REM["強度<br/>(remission)"]
        NORM["法線<br/>(nx, ny, nz)"]
        GT["GTラベル<br/>(class)"]
    end
    
    subgraph Stage1["init SP"]
        S1["RANSAC + DBSCAN"]
    end
    
    subgraph Stage2["Backbone"]
        S2["Res16FPN18"]
    end
    
    subgraph Stage3["SP Constructor"]
        S3["k-means + 特徴連結"]
    end
    
    subgraph Stage4["SPC"]
        S4["k-means → Classifier"]
    end
    
    XYZ -->|"✅"| S1
    XYZ -->|"✅"| S2
    
    REM -->|"❌"| S1
    REM -->|"❌"| S2
    REM -->|"✅"| S3
    
    NORM -->|"❌"| S1
    NORM -->|"❌"| S2
    NORM -->|"✅ PFH"| S3
    
    GT -->|"❌ (評価のみ)"| S1
    GT -->|"❌"| S2
    GT -->|"❌"| S3
    GT -->|"❌"| S4
    
    S1 -->|"SPラベル"| S3
    S2 -->|"点特徴128d"| S3
    S3 -->|"SP特徴139d"| S4

    style XYZ fill:#bbdefb
    style REM fill:#ffe0b2
    style NORM fill:#c5e1a5
    style GT fill:#ffcdd2
```

---

## ハイパーパラメータ一覧

| 処理 | パラメータ | デフォルト値 | 説明 |
|------|-----------|-------------|------|
| **init SP** | `distance_threshold` | 0.1 | RANSAC距離閾値 (m) |
| | DBSCAN `eps` | 0.2 | クラスタリング距離 (m) |
| | DBSCAN `min_points` | 1 | 最小点数 |
| **Backbone** | `input_dim` | 3 | 入力次元 (座標のみ) |
| | `feats_dim` | 128 | 出力特徴次元 |
| | `voxel_size` | 0.15 | Voxelサイズ (m) |
| | `conv1_kernel_size` | 5 | 初期畳み込みカーネル |
| | `bn_momentum` | 0.02 | BatchNormモメンタム |
| **SP Constructor** | `growsp.start` | 80 | 初期SP数 |
| | `growsp.end` | 30 | 最終SP数 |
| | `growsp.c_rgb` | 5.0 | 強度特徴の重み |
| | `growsp.c_shape` | 5.0 | PFH特徴の重み |
| **SPC** | `primitive_num` | 500 | Primitive数 |
| | `select_num` | 1500 | クラスタリング用シーン数 |



