"""
    Modified from https://github.com/XMUDeepLIT/SSR/blob/main/custom/icl_gen/kmeans_self.py.
    Hyperparameters are taken from https://arxiv.org/abs/2403.01244.
"""


import argparse
import os
import numpy as np
import jsonlines
import random
from sklearn.cluster import KMeans

np.random.seed(0)
random.seed(0)

def main(config):
    embedding_path = config['embedding_path']
    generated_path = config['generated_path']
    output_path = config['output_path']
    os.makedirs(output_path, exist_ok=True)

    n_cluster = config['n_cluster']

    for fname in os.listdir(embedding_path):
        if fname.endswith('.npy'):
            print(f"Processing task: {fname}")

            emb_path = os.path.join(embedding_path, fname)
            embeddings = np.load(emb_path)
            n_emb, n_dim = embeddings.shape

            # KMeans clustering
            kmeans = KMeans(n_clusters=n_cluster, n_init='auto')
            labels = kmeans.fit_predict(embeddings)
            centric_distances = np.array([np.linalg.norm(e-kmeans.cluster_centers_[labels[i]]) for i, e in enumerate(embeddings)])

            n_cluster_instances = [0]* n_cluster
            uniq_idx, uniq_cnt = np.unique(labels, return_counts=True)
            for i, idx in enumerate(uniq_idx):
                n_cluster_instances[idx] = uniq_cnt[i]

            clu_sample_num = [round(config["sample_memory"]*n/n_emb) for n in n_cluster_instances]

            fname_json = fname.replace('.npy', '.json')
            gen_path = os.path.join(generated_path, fname_json)
            with jsonlines.open(gen_path) as f:
                raw_data = [l for l in f]

            sampled_data = []
            for clu_idx in range(n_cluster):
                cur_clu_idx_list = np.where(labels==clu_idx)[0]
                cur_clu_dis_list = centric_distances[cur_clu_idx_list]
                easys = np.argsort(cur_clu_dis_list)[:clu_sample_num[clu_idx]]

                for samp_idx in easys:
                    sampled_data.append(raw_data[cur_clu_idx_list[samp_idx]])

            print("Original size:", len(raw_data))
            print("Processed size:", len(sampled_data))

            output_file = os.path.join(output_path, fname_json)
            with jsonlines.open(output_file, 'w') as f:
                f.write_all(sampled_data)

def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_stream", type=str, default="CITB")
    parser.add_argument("--embedding_path", type=str, default="callm/data/citb/generated_data/CITB/text2emb_v0.6_new")
    parser.add_argument("--generated_path", type=str, default="callm/data/citb/generated_data/CITB/v0.6_remove_duplicates")
    parser.add_argument("--output_path", type=str, default="callm/data/citb/generated_data/CITB/v0.6_kmeans_postprocess")
    parser.add_argument("--sample_memory", type=int, default=3000)
    parser.add_argument("--n_cluster", type=int, default=20)

    args = parser.parse_args()
    config = vars(args)

    return config

if __name__ == "__main__":
    config = get_config()
    main(config)
