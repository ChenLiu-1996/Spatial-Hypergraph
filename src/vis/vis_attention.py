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
def save_test_set_attentions(model, test_loader, device, attention_save_path):
    attention_arr, subject_label_arr = None, None

    # Aggregate the MLP weights.
    linear_layers = [m for m in model.classifier.modules() if isinstance(m, torch.nn.Linear)]
    W = linear_layers[-1].weight
    for layer in linear_layers[:-1][::-1]:
        W = W @ layer.weight  # Chain multiplication
    mlp_weights = W.squeeze()  # [num_classes, wavelet_scales * num_features]

    for data_item in tqdm(test_loader):
        data_item = data_item.to(device)
        y_true = torch.Tensor(data_item.y).long().to(device)
        attention_weights = model(
            x=data_item.x,
            hyperedge_index=data_item.edge_index,
            hyperedge_attr=data_item.edge_attr,
            batch=data_item.batch,
            return_attention=True)

        attention_weights = attention_weights.cpu().detach().numpy().astype(np.float16)
        subject_label = y_true.cpu().detach().numpy().reshape(-1, 1)

        if attention_arr is None:
            attention_arr = attention_weights
            subject_label_arr = subject_label
        else:
            attention_arr = np.concatenate((attention_arr, attention_weights), axis=0)
            subject_label_arr = np.concatenate((subject_label_arr, subject_label), axis=0)

    with open(attention_save_path, 'wb+') as f:
        np.savez(f,
                 attention_arr=attention_arr,
                 subject_label_arr=subject_label_arr,
                 mlp_weights=mlp_weights)
    return

def visualize_test_set_attention(embedding_save_path, class_map):
    with open(embedding_save_path, 'rb') as f:
        npzfile = np.load(f)
        attention_arr = npzfile['attention_arr']
        subject_label_arr = npzfile['subject_label_arr']
        mlp_weights = npzfile['mlp_weights']

        fig = plt.figure(figsize=(24, 16))
        for class_idx, class_name in class_map.items():
            subject_indices = (subject_label_arr == class_idx).flatten()

            attention_curr_class = attention_arr[subject_indices, ...].mean(axis=0)
            attention_curr_class = (attention_curr_class - attention_curr_class.min()) / (
                attention_curr_class.max() - attention_curr_class.min())

            ax = fig.add_subplot(2, len(class_map.items()), class_idx + 1)
            matrix_fig = ax.imshow(attention_curr_class, cmap='coolwarm')
            ax.set_xticks([0, 63, 127, 191, 255])
            ax.set_xticklabels([0, 63, 127, 191, 255])
            ax.set_yticks([0, 63, 127, 191, 255])
            ax.set_yticklabels([0, 63, 127, 191, 255])
            ax.set_title(class_name, fontsize=16)
            cbar = fig.colorbar(matrix_fig, ax=ax)
            ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(ticks)

            if class_idx == 0:
                ax.set_ylabel('Attention weights only', fontsize=16)

        fig.tight_layout(pad=2)
        fig.savefig(os.path.join(os.path.dirname(embedding_save_path), 'attentions.png'))

        for class_idx, class_name in class_map.items():
            subject_indices = (subject_label_arr == class_idx).flatten()

            attention_curr_class = attention_arr[subject_indices, ...].mean(axis=0)
            mlp_weight_curr_class = mlp_weights[class_idx, :]
            attention_curr_class = attention_curr_class * mlp_weight_curr_class[None, :]
            attention_curr_class = (attention_curr_class - attention_curr_class.min()) / (
                attention_curr_class.max() - attention_curr_class.min())

            ax = fig.add_subplot(2, len(class_map.items()), len(class_map.items()) + class_idx + 1)
            matrix_fig = ax.imshow(attention_curr_class, cmap='coolwarm')
            ax.set_xticks([0, 63, 127, 191, 255])
            ax.set_xticklabels([0, 63, 127, 191, 255])
            ax.set_yticks([0, 63, 127, 191, 255])
            ax.set_yticklabels([0, 63, 127, 191, 255])
            ax.set_title(class_name, fontsize=16)
            cbar = fig.colorbar(matrix_fig, ax=ax)
            ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(ticks)

            if class_idx == 0:
                ax.set_ylabel('Attention weights scaled by MLP weights', fontsize=16)

        fig.tight_layout(pad=2)
        fig.savefig(os.path.join(os.path.dirname(embedding_save_path), 'attentions.png'))
        plt.close(fig)
    return


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Entry point.')
    args.add_argument('--train-val-test-ratio', default='6:2:2', type=str)
    args.add_argument('--trainable-scales', action='store_true')
    args.add_argument('--k-hop', default=1, type=int)
    args.add_argument('--num-workers', default=8, type=int)
    args.add_argument('--random-seed', default=1, type=int)
    args.add_argument('--dataset', default='placenta', type=str)
    args.add_argument('--data-folder', default='$ROOT/data/spatial_placenta_accreta/patchified/', type=str)
    args.add_argument('--num-features', default=18085, type=int)  # number of genes or features

    args = args.parse_known_args()[0]
    args.batch_size = 1
    seed_everything(args.random_seed)

    # Update paths with absolute path.
    args.data_folder = args.data_folder.replace('$ROOT', ROOT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the data.
    train_loader, val_loader, test_loader, dataset = prepare_dataloaders(args)

    model = HypergraphScatteringNet(
        in_channels=64,
        hidden_channels=16,
        out_channels=dataset.num_classes,
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

    subset_name = os.path.basename(args.data_folder.rstrip('/'))
    current_run_identifier = f'dataset-{args.dataset}-{subset_name}_kHop-{args.k_hop}_features-{args.num_features}_trainable_scales-{args.trainable_scales}_seed-{args.random_seed}'
    model_save_path = os.path.join(ROOT_DIR, 'results', args.dataset, current_run_identifier, 'model.pt')
    attention_save_path = os.path.join(ROOT_DIR, 'results', args.dataset, current_run_identifier, 'attentions.npz')
    os.makedirs(os.path.dirname(attention_save_path), exist_ok=True)

    model.eval()
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    if not os.path.isfile(attention_save_path):
        save_test_set_attentions(model, test_loader, device, attention_save_path)

    class_map = test_loader.dataset.dataset.class_map
    visualize_test_set_attention(attention_save_path, class_map)