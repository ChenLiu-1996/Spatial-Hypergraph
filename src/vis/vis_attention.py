import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
from matplotlib import pyplot as plt

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])
sys.path.insert(0, import_dir + '/src/utils/')
from seed import seed_everything

sys.path.insert(0, import_dir + '/src/')
from model.hypergraph_scattering import HypergraphScatteringNet
from train import prepare_dataloaders

ROOT_DIR = '/'.join(os.path.realpath(__file__).split('/')[:-3])


@torch.no_grad()
def save_test_set_attentions(model, test_loader, device, attention_save_path):
    niche_attention_list, coords_list = [], []
    feature_attention_arr, y_true_arr, y_pred_arr = None, None, None

    # Aggregate the MLP weights.
    linear_layers = [m for m in model.classifier.modules() if isinstance(m, torch.nn.Linear)]
    W = linear_layers[-1].weight
    for layer in linear_layers[:-1][::-1]:
        W = W @ layer.weight  # Chain multiplication
    mlp_weights = W.squeeze()  # [num_classes, wavelet_scales * num_features]

    for data_item in tqdm(test_loader):
        data_item = data_item.to(device)
        y_true = torch.Tensor(data_item.y).long().to(device)
        niche_attn, feature_attn = model(
            x=data_item.x,
            hyperedge_index=data_item.edge_index,
            hyperedge_attr=data_item.edge_attr,
            batch=data_item.batch,
            return_attention=True)
        y_pred = model(
            x=data_item.x,
            hyperedge_index=data_item.edge_index,
            hyperedge_attr=data_item.edge_attr,
            batch=data_item.batch)

        assert len(niche_attn.shape) == 1
        niche_attn = niche_attn.cpu().detach().numpy().astype(np.float16)
        feature_attn = feature_attn.cpu().detach().numpy().astype(np.float16)
        y_true = y_true.cpu().detach().numpy().reshape(1, 1)
        y_pred = y_pred.cpu().detach().numpy().reshape(1, -1)
        assert len(data_item.coords) == 1
        coords = data_item.coords[0]

        if feature_attention_arr is None:
            niche_attention_list = [niche_attn]
            coords_list = [coords]
            feature_attention_arr = feature_attn
            y_true_arr = y_true
            y_pred_arr = y_pred
        else:
            niche_attention_list.append(niche_attn)
            coords_list.append(coords)
            feature_attention_arr = np.concatenate((feature_attention_arr, feature_attn), axis=0)
            y_true_arr = np.concatenate((y_true_arr, y_true), axis=0)
            y_pred_arr = np.concatenate((y_pred_arr, y_pred), axis=0)

    with open(attention_save_path, 'wb+') as f:
        np.savez(f,
                 niche_attention_arr=np.array(niche_attention_list, dtype=object),
                 coords_arr=np.array(coords_list, dtype=object),
                 feature_attention_arr=feature_attention_arr,
                 y_true_arr=y_true_arr,
                 y_pred_arr=y_pred_arr,
                 mlp_weights=mlp_weights)
    return


def visualize_test_set_feature_attention(embedding_save_path, gene_list, class_map):
    npzfile = np.load(embedding_save_path, allow_pickle=True)
    feature_attention_arr = npzfile['feature_attention_arr']
    y_true_arr = npzfile['y_true_arr']
    mlp_weights = npzfile['mlp_weights']

    fig = plt.figure(figsize=(24, 16))
    for class_idx, class_name in class_map.items():
        subject_indices = (y_true_arr == class_idx).flatten()

        attention_curr_class = feature_attention_arr[subject_indices, ...].mean(axis=0)
        attention_curr_class = (attention_curr_class - attention_curr_class.min()) / (
            attention_curr_class.max() - attention_curr_class.min())

        ax = fig.add_subplot(2, len(class_map.items()), class_idx + 1)
        matrix_fig = ax.imshow(attention_curr_class, cmap='coolwarm')
        ax.set_title(class_name, fontsize=16)
        cbar = fig.colorbar(matrix_fig, ax=ax)
        ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticks)

        if class_idx == 0:
            ax.set_ylabel('Attention weights only', fontsize=16)

    fig.tight_layout(pad=2)
    fig.savefig(os.path.join(os.path.dirname(embedding_save_path), 'feature_attentions.png'), dpi=200)

    for class_idx, class_name in class_map.items():
        subject_indices = (y_true_arr == class_idx).flatten()

        attention_curr_class = feature_attention_arr[subject_indices, ...].mean(axis=0)
        mlp_weight_curr_class = mlp_weights[class_idx, :]
        attention_curr_class = attention_curr_class * mlp_weight_curr_class[None, :]
        attention_curr_class = (attention_curr_class - attention_curr_class.min()) / (
            attention_curr_class.max() - attention_curr_class.min())

        ax = fig.add_subplot(2, len(class_map.items()), len(class_map.items()) + class_idx + 1)
        matrix_fig = ax.imshow(attention_curr_class, cmap='coolwarm')
        ax.set_title(class_name, fontsize=16)
        cbar = fig.colorbar(matrix_fig, ax=ax)
        ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticks)

        if class_idx == 0:
            ax.set_ylabel('Attention weights scaled by MLP weights', fontsize=16)

    fig.tight_layout(pad=2)
    fig.savefig(os.path.join(os.path.dirname(embedding_save_path), 'feature_attentions.png'))
    plt.close(fig)

    fig = plt.figure(figsize=(12 * len(class_map), feature_attention_arr.shape[1]/4))
    for class_idx, class_name in class_map.items():
        subject_indices = (y_true_arr == class_idx).flatten()

        attention_curr_class = feature_attention_arr[subject_indices, ...].mean(axis=0)
        attention_curr_class = (attention_curr_class - attention_curr_class.min()) / (
            attention_curr_class.max() - attention_curr_class.min())

        feature_importance = attention_curr_class.sum(axis=0)
        # Sort descendingly.
        sorted_idx = np.argsort(feature_importance)
        importance_sorted = feature_importance[sorted_idx]
        gene_names_sorted = np.array(gene_list)[sorted_idx]

        ax = fig.add_subplot(1, 2 * len(class_map.items()), class_idx + 1)
        ax.barh(range(len(importance_sorted)), importance_sorted, color='firebrick', alpha=0.5)
        ax.set_title(class_name, fontsize=24)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim([0, len(gene_names_sorted)])
        ax.set_yticks(range(len(gene_names_sorted)))
        ax.set_yticklabels(gene_names_sorted)
        ax.tick_params(axis='x', which='major', labelsize=18)

        if class_idx == 0:
            ax.set_ylabel('Attention weights only', fontsize=16)

    fig.tight_layout(pad=2)
    fig.savefig(os.path.join(os.path.dirname(embedding_save_path), 'feature_importance.png'))

    for class_idx, class_name in class_map.items():
        subject_indices = (y_true_arr == class_idx).flatten()

        attention_curr_class = feature_attention_arr[subject_indices, ...].mean(axis=0)
        mlp_weight_curr_class = mlp_weights[class_idx, :]
        attention_curr_class = attention_curr_class * mlp_weight_curr_class[None, :]
        attention_curr_class = (attention_curr_class - attention_curr_class.min()) / (
            attention_curr_class.max() - attention_curr_class.min())

        feature_importance = attention_curr_class.sum(axis=0)
        # Sort descendingly.
        sorted_idx = np.argsort(feature_importance)
        importance_sorted = feature_importance[sorted_idx]
        gene_names_sorted = np.array(gene_list)[sorted_idx]

        ax = fig.add_subplot(1, 2 * len(class_map.items()), len(class_map.items()) + class_idx + 1)
        ax.barh(range(len(importance_sorted)), importance_sorted, color='firebrick', alpha=0.5)
        ax.set_title(class_name, fontsize=24)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim([0, len(gene_names_sorted)])
        ax.set_yticks(range(len(gene_names_sorted)))
        ax.set_yticklabels(gene_names_sorted)
        ax.tick_params(axis='x', which='major', labelsize=18)

        if class_idx == 0:
            ax.set_ylabel('Attention weights scaled by MLP weights', fontsize=16)

    fig.tight_layout(pad=2)
    fig.savefig(os.path.join(os.path.dirname(embedding_save_path), 'feature_importance.png'), dpi=200)
    plt.close(fig)

    return


def visualize_test_set_niche_attention(embedding_save_path, gene_list, class_map, topk):
    npzfile = np.load(embedding_save_path, allow_pickle=True)
    coords_arr = npzfile['coords_arr']
    niche_attention_arr = npzfile['niche_attention_arr']
    y_true_arr = npzfile['y_true_arr']
    y_pred_arr = npzfile['y_pred_arr']

    # Softmax over classes.
    y_pred_arr = softmax(y_pred_arr)
    confidence_arr = y_pred_arr.max(axis=1).reshape(-1, 1)
    y_pred_label_arr = y_pred_arr.argmax(axis=1).reshape(-1, 1)

    fig = plt.figure(figsize=(12 * len(class_map), 8 * topk))
    for class_idx, class_name in class_map.items():
        # Correct prediction, confidence, and indices.
        mask_correct = (y_true_arr == class_idx) & (y_pred_label_arr == class_idx)
        confidence_correct = confidence_arr[mask_correct]
        idx_correct = np.flatnonzero(mask_correct)

        # Find topk most confident ones among correct predictions.
        assert len(confidence_correct) > 0
        topk_correct = idx_correct[np.argsort(-confidence_correct)[:topk]]

        for subplot_idx, hypergraph_idx in enumerate(topk_correct):
            ax = fig.add_subplot(len(class_map), topk * 2, class_idx * topk * 2 + subplot_idx + 1)
            ax.set_title(class_name + f'\nTop{subplot_idx+1} confidence\nCORRECT prediction', fontsize=24)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=18)

            coords = coords_arr[hypergraph_idx]
            niche_attention = niche_attention_arr[hypergraph_idx]

            scatter = ax.scatter(coords[:, 0], coords[:, 1], s=500, c=niche_attention, cmap='coolwarm')
            fig.colorbar(scatter, ax=ax)

        # Correct prediction, confidence, and indices.
        mask_incorrect = ~mask_correct
        confidence_incorrect = confidence_arr[mask_incorrect]
        idx_incorrect = np.flatnonzero(mask_incorrect)

        # Find topk most confident ones among correct predictions.
        assert len(confidence_incorrect) > 0
        topk_incorrect = idx_incorrect[np.argsort(-confidence_incorrect)[:topk]]

        for subplot_idx, hypergraph_idx in enumerate(topk_incorrect):
            ax = fig.add_subplot(len(class_map), topk * 2, class_idx * topk * 2 + topk + subplot_idx + 1)
            ax.set_title(class_name + f'\nTop{subplot_idx+1} confidence\nINCORRECT prediction', fontsize=24)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=18)

            coords = coords_arr[hypergraph_idx]
            niche_attention = niche_attention_arr[hypergraph_idx]

            scatter = ax.scatter(coords[:, 0], coords[:, 1], s=500, c=niche_attention, cmap='coolwarm')
            fig.colorbar(scatter, ax=ax)

    fig.tight_layout(pad=2)
    fig.savefig(os.path.join(os.path.dirname(embedding_save_path), 'topK_bottomK_niches.png'), dpi=200)
    plt.close(fig)

    return


def softmax(x):
# Subtract max for numerical stability
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=-1, keepdims=True)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Entry point.')
    args.add_argument('--train-val-test-ratio', default='6:2:2', type=str)
    args.add_argument('--trainable-scales', action='store_true')
    args.add_argument('--k-hop', default=3, type=int)
    args.add_argument('--num-workers', default=8, type=int)
    args.add_argument('--random-seed', default=1, type=int)
    args.add_argument('--dataset', default='placenta', type=str)
    args.add_argument('--data-folder', default='$ROOT/data/spatial_placenta_accreta/patchified_selected_genes', type=str)
    args.add_argument('--num-features', default=212, type=int)  # number of genes or features
    args.add_argument('--topk', default=3, type=int, help='for niche attention visualization')

    args = args.parse_known_args()[0]
    args.batch_size = 1
    seed_everything(args.random_seed)

    # Update paths with absolute path.
    args.data_folder = args.data_folder.replace('$ROOT', ROOT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the data.
    train_loader, val_loader, test_loader, num_classes = prepare_dataloaders(args)

    model = HypergraphScatteringNet(
        in_channels=64,
        hidden_channels=16,
        out_channels=num_classes,
        num_features=args.num_features,
        trainable_laziness=False,
        trainable_scales=args.trainable_scales,
        activation=None,  # just get one layer of wavelet transform
        fixed_weights=True,
        layout=['hsm'],
        normalize='right',
        pooling='attention',
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

    gene_list = test_loader.dataset.dataset.gene_list
    class_map = test_loader.dataset.dataset.class_map
    visualize_test_set_feature_attention(attention_save_path, gene_list=gene_list, class_map=class_map)
    visualize_test_set_niche_attention(attention_save_path, gene_list=gene_list, class_map=class_map, topk=args.topk)
