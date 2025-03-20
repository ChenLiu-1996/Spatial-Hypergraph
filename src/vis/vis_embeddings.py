import argparse
import os
import sys
import meld
import phate
import scprep
import numpy as np
from tqdm import tqdm
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])
sys.path.insert(0, import_dir + '/src/utils/')
from seed import seed_everything

sys.path.insert(0, import_dir + '/src/')
from model.hypergraph_scattering import HypergraphScatteringNet
from train import prepare_dataloaders

ROOT_DIR = '/'.join(os.path.realpath(__file__).split('/')[:-3])

@torch.no_grad()
def save_test_set_embeddings(model, test_loader, device, embedding_save_path):
    hyperedge_emb_arr, hyperedge_label_arr = None, None
    hypergraph_emb_arr, hypergraph_label_arr = None, None

    for data_item in tqdm(test_loader):
        data_item = data_item.to(device)
        y_true = torch.Tensor(data_item.y).long().to(device)
        embedding = model(
            x=data_item.x,
            hyperedge_index=data_item.edge_index,
            hyperedge_attr=data_item.edge_attr,
            batch=data_item.batch,
            return_wavelet_embeddings=True)

        hyperedge_emb = embedding.cpu().detach().numpy()
        # `hyperedge_emb.shape[0]` is number of hyperedges. Repeat the label for each hyperedge.
        hyperedge_label = y_true.cpu().detach().numpy().reshape(-1, 1).repeat(hyperedge_emb.shape[0], axis=0)
        hypergraph_emb = embedding.cpu().detach().numpy().mean(axis=0, keepdims=True)
        hypergraph_label = y_true.cpu().detach().numpy().reshape(-1, 1)

        if hypergraph_emb_arr is None:
            hyperedge_emb_arr = hyperedge_emb
            hyperedge_label_arr = hyperedge_label
            hypergraph_emb_arr = hypergraph_emb
            hypergraph_label_arr = hypergraph_label
        else:
            hyperedge_emb_arr = np.concatenate((hyperedge_emb_arr, hyperedge_emb), axis=0)
            hyperedge_label_arr = np.concatenate((hyperedge_label_arr, hyperedge_label), axis=0)
            hypergraph_emb_arr = np.concatenate((hypergraph_emb_arr, hypergraph_emb), axis=0)
            hypergraph_label_arr = np.concatenate((hypergraph_label_arr, hypergraph_label), axis=0)

    with open(embedding_save_path, 'wb+') as f:
        np.savez(f,
                 hyperedge_emb_arr=hyperedge_emb_arr, hyperedge_label_arr=hyperedge_label_arr,
                 hypergraph_emb_arr=hypergraph_emb_arr, hypergraph_label_arr=hypergraph_label_arr)
    return

def visualize_test_set_embeddings(embedding_save_path, class_map):
    with open(embedding_save_path, 'rb') as f:
        npzfile = np.load(f)
        hyperedge_label_arr = npzfile['hyperedge_label_arr']
        hyperedge_emb_arr = npzfile['hyperedge_emb_arr']
        hypergraph_emb_arr = npzfile['hypergraph_emb_arr']
        hypergraph_label_arr = npzfile['hypergraph_label_arr']

    # NOTE: Plot visualization where each node is a hypergraph (figure 1) or a hyperedge (figure 2).
    for node_rep, file_name, embedding_label_pair in \
        zip(['Each node is a hypergraph.', 'Each node is a hyperedge.'],
            ['hypergraph_embeddings.png', 'hyperedge_embeddings.png'],
            [(hypergraph_emb_arr, hypergraph_label_arr), (hyperedge_emb_arr, hyperedge_label_arr)]):

        embedding_arr, label_arr = embedding_label_pair

        phate_op = phate.PHATE(random_state=args.random_seed, n_jobs=args.num_workers, verbose=True)
        data_phate = phate_op.fit_transform(normalize(embedding_arr, axis=1))

        fig = plt.figure(figsize=(12, 8))
        for class_idx, class_name in class_map.items():
            ax = fig.add_subplot(2, len(class_map.items()), class_idx + 1)
            color_arr = label_arr == class_idx
            scprep.plot.scatter2d(
                data_phate,
                c=color_arr,
                cmap='coolwarm',
                ax=ax,
                title=class_name,
                xticks=False,
                yticks=False,
                label_prefix='PHATE',
                fontsize=10,
                s=5,
                alpha=0.5)
        fig.suptitle(node_rep)
        fig.tight_layout(pad=2)
        fig.savefig(os.path.join(os.path.dirname(embedding_save_path), file_name))

        meld_op = meld.MELD(random_state=args.random_seed, n_jobs=args.num_workers, verbose=True)
        sample_densities = meld_op.fit_transform(normalize(embedding_arr, axis=1), sample_labels=label_arr)
        sample_likelihoods = meld.utils.normalize_densities(sample_densities)
        for class_idx, class_name in class_map.items():
            ax = fig.add_subplot(2, len(class_map.items()), len(class_map.items()) + class_idx + 1)
            color_arr = sample_likelihoods[label_arr == class_idx]
            scprep.plot.scatter2d(
                data_phate,
                c=color_arr,
                cmap='coolwarm',
                ax=ax,
                title=class_name+' (MELD)',
                xticks=False,
                yticks=False,
                label_prefix='PHATE',
                fontsize=10,
                s=5,
                alpha=0.5)
        fig.tight_layout(pad=2)
        fig.savefig(os.path.join(os.path.dirname(embedding_save_path), file_name))
        plt.close(fig)
    return


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Entry point.')
    args.add_argument('--train-val-test-ratio', default='6:2:2', type=str)
    args.add_argument('--batch-size', default=1, type=int)
    args.add_argument('--trainable-scales', action='store_true')
    args.add_argument('--num-workers', default=8, type=int)
    args.add_argument('--random-seed', default=1, type=int)
    args.add_argument('--data-folder', default='$ROOT/data/spatial_placenta_accreta/patchified/', type=str)
    args.add_argument('--num-features', default=18085, type=int)  # number of genes or features

    args = args.parse_known_args()[0]
    seed_everything(args.random_seed)

    # Update paths with absolute path.
    args.data_folder = args.data_folder.replace('$ROOT', ROOT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HypergraphScatteringNet(
        in_channels=64,
        hidden_channels=16,
        out_channels=3,
        num_features=args.num_features,
        trainable_laziness=False,
        trainable_scales=args.trainable_scales,
        activation=None,  # just get one layer of wavelet transform
        fixed_weights=True,
        layout=['hsm'],
        normalize='right',
        pooling='mean',
        scale_list=[0,1,2,4]
    )
    model.eval()
    model.to(device)

    # Load the data.
    train_loader, val_loader, test_loader = prepare_dataloaders(args)

    model_save_path = os.path.join(ROOT_DIR, 'results', f'model_features-{args.num_features}_trainable_scales-{args.trainable_scales}_seed-{args.random_seed}.pt')
    embedding_save_path = os.path.join(ROOT_DIR, 'results', 'embeddings', f'model_features-{args.num_features}_trainable_scales-{args.trainable_scales}_seed-{args.random_seed}', 'embeddings.npz')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(embedding_save_path), exist_ok=True)

    model.eval()
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    if not os.path.isfile(embedding_save_path):
        save_test_set_embeddings(model, test_loader, device, embedding_save_path)

    class_map = test_loader.dataset.dataset.class_map
    visualize_test_set_embeddings(embedding_save_path, class_map)