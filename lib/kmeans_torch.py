# MIT License

# Copyright (c) 2021 densechen

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# from https://github.com/densechen/kmeans-gpu

from dis import dis
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class KMeans(nn.Module):
    r"""KMeans module with PyTorch support.

    Args:
        n_clusters: Number of clusters.
        max_iter: Maximum number of iterations.
        tolerance: Tolerance for error/distance.

        distance: `euclidean` or `cosine`.
        sub_sampling: The number of points used in KMeans.
            If None, use all points to do KMeans.
        max_neighbors: The number of neighbors to use for aggregating features.
    """

    def __init__(self,
                 n_clusters: int,
                 max_iter: int = 100,
                 tolerance: float = 1e-4,
                 distance: str = 'euclidean',
                 sub_sampling: int = None,
                 max_neighbors: int = 15,
                 differentiable: int = False):
        super().__init__()
        assert distance in ['euclidean', 'cosine']
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.distance = distance
        self.sub_sampling = sub_sampling
        self.max_neighbors = max_neighbors
        self.differentiable = differentiable

    @classmethod
    def cos_sim(cls, vec_a, vec_b):
        """Compute Cosine Similarity between vec_a and vec_b.
        Args:
            vec_a: m x d
            vec_b: n x d

        Returns:
            m x n
        """
        vec_a = vec_a.unsqueeze(1).expand(vec_a.shape[0], vec_b.shape[0], -1)
        vec_b = vec_b.unsqueeze(0).expand_as(vec_a)
        return F.cosine_similarity(vec_a, vec_b, dim=-1)

    @classmethod
    def euc_sim(cls, vec_a, vec_b):
        r"""Compute Euclidean Distance between vec_a and vec_b.
        Args:
            vec_a: m x d
            vec_b: n x d

        Returns:
            m x n
        """
        vec_a = vec_a.to(dtype=torch.float64)
        vec_b = vec_b.to(dtype=torch.float64)
        # (vec_a - vec_b)^2 = vec_a^2 + vec_b.T^2 - 2 vec_a @ vec_b.T
        return (2 * vec_a @ vec_b.T - (vec_a**2).sum(dim=1, keepdim=True) - (vec_b.T**2).sum(dim=0, keepdim=True))

    @classmethod
    def max_sim(cls, vec_a, vec_b, distance):
        """Compute maximum similarity (or minimum distance) of each vector in vec_a with all of the vectors in vec_b.

        Args:
            vec_a: m x d
            vec_b: n x d
        Returns:
            [value, indices]: m
        """
        sim_score = KMeans.cos_sim(
            vec_a, vec_b) if distance == "cosine" else KMeans.euc_sim(vec_a, vec_b)
        return sim_score.max(dim=-1)

    @classmethod
    def predict(cls, X, centroids, distance):
        """Predict the closest cluster each sample in X belongs to.
        Args:
            X: n x d
            centroids: m x d
            distance:

        Returns:
            labels: n
        """
        return cls.max_sim(vec_a=X, vec_b=centroids, distance=distance)[1]

    def kmeans_plusplus(self, X):
        if self.distance == "cosine":
            raise NotImplementedError("Cosine distance is not supported for KMeans++")
        n_samples, _ = X.shape
        centroids = []
        # 最初のセントロイドをランダムに選択
        idx = torch.randint(0, n_samples, (1,))
        centroids.append(X[idx])

        for i in range(1, self.n_clusters):
            distances = torch.stack([KMeans.euc_sim(X, c) for c in centroids])

            distances *= -1.0
            min_distances, _ = distances.min(dim=0)
            probabilities = min_distances.squeeze() / min_distances.sum()
            probabilities = torch.clamp(probabilities, min=0.0)
            idx = torch.multinomial(probabilities, 1)
            centroids.append(X[idx])

        return torch.cat(centroids, dim=0)

    @torch.no_grad()
    def fit_predict(self, X, centroids=None):
        """Combination of fit() and predict() methods.
        Args:
            X: torch.Tensor, shape: [n_samples, n_features]
            centroids: {torch.Tensor, None}, default: None
                If given, centroids will be initialized with given tensor
                If None, centroids will be randomly chosen from X
            Return:
                labels: n_samples
                centroids: n_samples x 3
        """
        pts, _ = X.shape
        device = X.device
        if centroids is None:
            a = np.random.choice(
                pts, size=[self.n_clusters], replace=False)
            centroids = X[a]
            # centroids = self.kmeans_plusplus(X)

        num_points_in_clusters = torch.ones(self.n_clusters, device=device)
        for _ in range(self.max_iter):
            # 1. Data propare
            if not self.sub_sampling:
                x = X
            else:
                # Sampling a subset to speedup KMeans
                x = X[np.random.choice(
                    pts, size=[self.sub_sampling], replace=False)]

            # 2. Similarity
            closest = KMeans.max_sim(vec_a=x, vec_b=centroids, distance=self.distance)[1]

            matched_clusters, counts = closest.unique(return_counts=True)

            c_grad = torch.zeros_like(centroids)
            matched_clusters_ = torch.arange(
                self.n_clusters, device=device) if not self.sub_sampling else matched_clusters
            expanded_closest = closest.unsqueeze(
                0).expand(len(matched_clusters_), -1)
            mask = (expanded_closest == matched_clusters_[:, None]).float()
            c_grad[matched_clusters_] = mask @ x / \
                (mask.sum(-1, keepdim=True) + 1e-8)

            error = (c_grad - centroids).pow(2).sum()
            lr = (
                0.9 / (num_points_in_clusters[:, None] + 1e-8) + 0.1) if self.sub_sampling else 1

            num_points_in_clusters[matched_clusters] += counts

            centroids = centroids * (1 - lr) + c_grad * lr
            if error <= self.tolerance:
                break
        if self.sub_sampling:
            closest = KMeans.predict(X, centroids, distance=self.distance)
        
        # 確認
        unique_clusters = torch.unique(closest)
        if len(unique_clusters) == self.n_clusters:
            return closest, centroids
        else:
            missing_clusters = set(range(self.n_clusters)) - set(unique_clusters.cpu().numpy())
            # print(f"Warning: Only {len(unique_clusters)} clusters found out of {self.n_clusters}")
            # print(f"Missing clusters: {sorted(missing_clusters)}")
            raise ValueError("KMeans failed to find all clusters")


    def single_batch_forward(self, points, features, centroids):
        """Actually, the KMeans process is not differentiable.
        Here, we make it as a differentiable process by using the weighted sum of cluster points.
        """
        closest, centroids = self.fit_predict(points, centroids)

        cluster_features = []
        cluster_centroids = []
        for cls in range(self.n_clusters):
            cp = points[closest == cls]

            # Compute distance to center points
            sim_score = KMeans.cos_sim(
                cp, centroids[cls:cls+1]) if self.distance == "cosine" else KMeans.euc_sim(cp, centroids[cls:cls+1])
            sim_score = sim_score.reshape(-1)
            score, index = torch.topk(sim_score, k=min(
                self.max_neighbors, len(cp)), largest=True)

            score = F.softmax(score, dim=0)

            # Select pts
            if self.differentiable:
                cp = cp[index, :]
                cluster_centroids.append(
                    torch.sum(cp * score.reshape(-1, 1), dim=0, keepdim=True))
            else:
                cluster_centroids.append(centroids[cls:cls+1])

            # Select features
            if features is not None:
                cf = features[:, closest == cls]
                cf = cf[:, index]
                cluster_features.append(
                    torch.sum(cf * score.reshape(1, -1), dim=1, keepdim=True))

        cluster_centroids = torch.cat(cluster_centroids, dim=0)
        if len(cluster_features) > 0:
            cluster_features = torch.cat(cluster_features, dim=1)
            return cluster_centroids, cluster_features, closest
        else:
            return cluster_centroids, None, closest

    def forward(self, points, features=None, centroids=None):
        r"""KMeans on points and then do an average aggregation on neighborhood points to get the feature for each cluster.
        Args:
            points: bz x n x 3
            features: bz x f x n, if features is given, we will aggregate the feature at the same time. 
            centroids: bz x m x 3, the initial centroids points.

        Returns:
            cluster centroids: bz x cc x 3
            cluster features: bz x f x cc
            cluster labels: bz x n
        """

        features = features if features is not None else [
            None for _ in range(len(points))]
        centroids = centroids if centroids is not None else [
            None for _ in range(len(points))]
        

        r_points, r_features, r_labels = [], [], []
        for pts, ft, ct in zip(points, features, centroids):
            pts, ft, lbl = self.single_batch_forward(pts, ft, ct)
            r_points.append(pts)
            r_features.append(ft)
            # print(points.shape, pts.shape, lbl.shape)
            r_labels.append(lbl)
        
        if features[0] is not None:
            return torch.stack(r_points, dim=0), torch.stack(r_features, dim=0), torch.stack(r_labels, dim=0)
        else:
            return torch.stack(r_points, dim=0), torch.stack(r_labels, dim=0)

if __name__ == '__main__':
    # seed 
    np.random.seed(0)
    torch.manual_seed(0)
    case = 4
    with torch.no_grad():
        if case == 1:
            model = KMeans(n_clusters=64).cuda()
            pcds = np.load('kmeans_error.npy')
            print(f"Input data shape: {pcds.shape}")
            pcds = torch.from_numpy(pcds).cuda().float()
            unsqueezed = pcds.unsqueeze(0)
        elif case == 2:
            model = KMeans(n_clusters=80).cuda()
            unsqueezed = torch.randn(4, 480000, 3).cuda() # [batch, points, features]
        elif case == 3:
            model = KMeans(n_clusters=80)
            unsqueezed = torch.randn(4, 480000, 3).cuda()
        elif case == 4:
            model = KMeans(n_clusters=80).cuda()
            unsqueezed = torch.randn(64, 480000, 3).cuda()
        elif case == 5:
            model = KMeans(n_clusters=80).cuda()
            unsqueezed = torch.randn(64, 480000, 3).cuda()
            # 並列処理のためにDataParallelを使用
            model = torch.nn.DataParallel(model)
        elif case == 6:
            from torch.multiprocessing import Process, set_start_method
            import time
            def kmeans_worker(data, gpu_id, results, idx):
                """個々のデータに対してKMeansを実行"""
                model = KMeans(n_clusters=80).cuda(gpu_id)
                data = data.cuda(gpu_id)
                start = time.time()
                centroids, labels = model(data)
                end = time.time()
                results[idx] = (centroids, labels, end - start)
                print(f"Worker {idx}: Completed in {end - start:.4f} seconds")

            def parallel_kmeans():
                set_start_method("spawn", force=True)  # マルチプロセスの初期化
                num_workers = 4
                gpu_id = 0  # 1つのGPUを使用
                data_chunks = [torch.randn(1, 480000, 3) for _ in range(num_workers)]  # 16個の独立データ
                results = [None] * num_workers

                # 各ワーカーを並列実行
                processes = []
                for i in range(num_workers):
                    p = Process(target=kmeans_worker, args=(data_chunks[i], gpu_id, results, i))
                    processes.append(p)
                    p.start()

                for p in processes:
                    p.join()

                # 結果とタイミングを確認
                total_time = sum([res[2] for res in results])
                print(f"Total parallel KMeans time: {total_time:.4f} seconds")
                return results
            start = time.time()
            results = parallel_kmeans()
            for idx, res in enumerate(results):
                centroids, labels, time = res
                print(f"Worker {idx}: Centroids shape: {centroids.shape}, Labels shape: {labels.shape}, Time: {time:.4f} seconds")
            print(f"Total parallel KMeans time: {time.time() - start:.4f} seconds")
            exit()
        elif case == 7:
            model = KMeans(n_clusters=80).cuda()
            unsqueezed = torch.randn(1, 480000, 3).cuda()
        import time
        start = time.time()
        centroids, labels = model(unsqueezed)
        print(f"KMeans GPU clustering completed in {time.time() - start:.4f} seconds")
        print(f"Centroids shape: {centroids.shape}, Labels shape: {labels.shape}")