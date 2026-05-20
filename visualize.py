import os
import hydra
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_util import ModelNet40, ScanObjectNN
from models.PointConT import PointConT_cls

@hydra.main(config_path='config', config_name='cls')
def visualize(args):
    print(f"Initializing Visualization for {args.dataset}...")
    
    # Setup device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    DATA_PATH = hydra.utils.to_absolute_path(args.dataset_dir)
    print(f"Loading {args.dataset} test dataset...")
    if args.dataset == 'ModelNet40':
        subset_fraction = args.get('subset_fraction', 1.0)
        test_dataset = ModelNet40(DATA_PATH, partition='test', num_points=args.num_points, subset_fraction=subset_fraction)
    elif args.dataset == 'ScanObjectNN':
        test_dataset = ScanObjectNN(DATA_PATH, partition='test', num_points=args.num_points)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} is not supported.")
        
    test_loader = DataLoader(test_dataset, num_workers=0, batch_size=8, shuffle=True, drop_last=False)
    
    # Determine the model checkpoint path
    # Since hydra changes the working directory, we need to locate the checkpoint.
    # We will look for it in the current run directory or the designated checkpoints dir.
    model_path = 'model.pth'
    if not os.path.exists(model_path):
        model_path = hydra.utils.to_absolute_path(f'checkpoints/{args.dataset}/{args.model_name}/{args.wandb_name}/model.pth')
    
    print(f'Loading model from {model_path} ...')
    model = PointConT_cls(args).to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except FileNotFoundError:
        print(f"Error: model.pth not found! Make sure you have trained the model first.")
        return
        
    model.eval()
    
    # Get one batch of data
    print("Running inference on a batch...")
    dataiter = iter(test_loader)
    data, labels = next(dataiter)
    data = data.to(device)
    labels = labels.squeeze()
    
    with torch.no_grad():
        logits = model(data)
        preds = logits.max(dim=1)[1].cpu().numpy()
        labels = labels.cpu().numpy()
        data = data.cpu().numpy()
        
    # Visualize
    vis_dir = hydra.utils.to_absolute_path('visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(f'{args.dataset} Point Cloud Classifications (Green=Correct, Red=Incorrect)', fontsize=16)
    
    for i in range(min(8, data.shape[0])):
        ax = fig.add_subplot(2, 4, i+1, projection='3d')
        pc = data[i]
        true_label = labels[i]
        pred_label = preds[i]
        
        color = 'green' if true_label == pred_label else 'red'
        
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=10, c=color, alpha=0.8)
        ax.set_title(f"True: {true_label} | Pred: {pred_label}", color=color, fontweight='bold')
        
        # Hide grid lines and axes to make it look cleaner
        ax.grid(False)
        ax.axis('off')
        
    plt.tight_layout()
    save_path = os.path.join(vis_dir, f'{args.dataset}_predictions.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nVisualization saved successfully to:\n{save_path}")

if __name__ == "__main__":
    visualize()
