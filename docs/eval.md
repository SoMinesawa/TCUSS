┌─────────────────────────────────────────────────────────────────────┐
│                        呼び出し元                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  【学習中の評価】                    【スタンドアロン評価】            │
│  train_SemanticKITTI.py             eval_SemanticKITTI.py (__main__)│
│         │                                    │                       │
│         ▼                                    ▼                       │
│  trainer.evaluate()                     eval()                       │
│  (lib/trainer.py)                  (eval_SemanticKITTI.py)          │
│         │                                    │                       │
│         │  ・モデル保存                       │  ・チェックポイント読込  │
│         │  ・KMeansで分類器作成              │  ・KMeansで分類器作成   │
│         │                                    │                       │
│         ▼                                    ▼                       │
│    eval_ddp()  ◀──────────────────────  eval_once()                 │
│  (eval_SemanticKITTI.py)            (eval_SemanticKITTI.py)          │
│         │                                                            │
│         │  ・DDP対応                                                  │
│         │  ・DistributedSampler使用                                   │
│         │  ・histogram集約                                            │
│         │                                                            │
│         ▼                                                            │
│    eval_once()                                                       │
│  (eval_SemanticKITTI.py)                                             │
│                                                                      │
│         │  ・DataLoaderをループ                                       │
│         │  ・model(in_field)で特徴抽出                                │
│         │  ・classifierで予測                                         │
│         │  ・preds, labels, distances, is_movingを返す                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘