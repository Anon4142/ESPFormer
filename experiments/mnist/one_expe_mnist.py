import torch
import torchvision
from vit import ViT_only_Att
import time
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--lr", type=float, default=0.002)
parser.add_argument("--attention_type", type=str, default='esp', choices=['esp', 'dif', 'vanilla', 'sink'])
parser.add_argument("--temperature", type=float, default=0.,
                    help="Temperature parameter for closed-form aggregation")
parser.add_argument("--gpu", type=int, default=0, help="GPU device to use (0, 1, 2, or 3)")
args = parser.parse_args()

# Parameters
seed = args.seed
lr = args.lr
attention_type = args.attention_type
agg_mode = args.agg_mode
temperature = args.temperature
device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

Dpath = 'mnist'
Bs_Train = 100
Bs_Test = 1000

# Data transforms and loaders
tform_mnist = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

tr_set = torchvision.datasets.MNIST(Dpath, train=True, download=True, transform=tform_mnist)
tr_load = torch.utils.data.DataLoader(tr_set, batch_size=Bs_Train, shuffle=True)

ts_set = torchvision.datasets.MNIST(Dpath, train=False, download=True, transform=tform_mnist)
ts_load = torch.utils.data.DataLoader(ts_set, batch_size=Bs_Test, shuffle=True)

# Training iteration
def train_iter(model, optimz, data_load, loss_val, save_adr):
    model.train()
    for i, (data, target) in enumerate(data_load):
        data = data.to(device)
        target = target.to(device)
        optimz.zero_grad()
        output, attn_weights = model(data)
        out = F.log_softmax(output, dim=1)
        loss = F.nll_loss(out, target)
        loss.backward()
        optimz.step()
        if i % 100 == 0:
            loss_val.append(loss.item())

# Evaluation
def evaluate(model, data_load, loss_val, test_acc):
    model.eval()
    samples = len(data_load.dataset)
    csamp = 0
    tloss = 0

    with torch.no_grad():
        for data, target in data_load:
            data = data.to(device)
            target = target.to(device)
            output = F.log_softmax(model(data)[0], dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)

            tloss += loss.item()
            csamp += pred.eq(target).sum()
    acc = 100.0 * csamp / samples
    aloss = tloss / samples
    loss_val.append(aloss)
    print('\nAverage test loss: {:.4f}  Accuracy: {}/{} ({:.2f}%)\n'.format(
        aloss, csamp, samples, acc))
    test_acc.append(acc.detach().cpu().item())

# Main training loop
def main(N_EPOCHS=45, heads=1, mlp_dim=128, lr=lr, depth=1,
         ps=4, seed=seed, save_adr='results_mnist', attention_type=attention_type,
         agg_mode=agg_mode, temperature=temperature):

    save_adr = f"{save_adr}_ps_{ps}_{attention_type}_{agg_mode}"
    os.makedirs(save_adr, exist_ok=True)

    #Shallow VIT
    model = ViT_only_Att(
        image_size=28, patch_size=ps, num_classes=10, channels=1,
        dim=128, depth=depth, heads=heads, mlp_dim=mlp_dim,
        attention_type=attention_type, interp=None, agg_mode=agg_mode, temperature=temperature
    ).to(device)
    optimz = optim.Adam(model.parameters(), lr=lr)

    trloss_val, tsloss_val, test_acc = [], [], []

    # Training loop
    for epoch in range(1, N_EPOCHS + 1):
        if epoch == 35 or epoch == 41:
            for g in optimz.param_groups:
                print('Reducing learning rate by 10')
                g['lr'] /= 10

        print('Epoch:', epoch)
        train_iter(model, optimz, tr_load, trloss_val, save_adr)
        evaluate(model, ts_load, tsloss_val, test_acc)

    # Save results
    torch.save(model.state_dict(), os.path.join(save_adr, 'final_model.pth'))
    np.save(os.path.join(save_adr, 'train_loss.npy'), np.array(trloss_val))
    np.save(os.path.join(save_adr, 'test_loss.npy'), np.array(tsloss_val))
    np.save(os.path.join(save_adr, 'test_accuracy.npy'), np.array(test_acc))

    print('Training complete. Results saved in:', save_adr)
    return test_acc

# Run experiments for different patch sizes
patch_sizes = [14, 28]
for ps in patch_sizes:
    print(f"Running model with patch size {ps}")
    main(N_EPOCHS=45, heads=1, mlp_dim=128, lr=lr, depth=1,
         ps=ps, seed=seed, save_adr='results_mnist',
         attention_type=attention_type, agg_mode=agg_mode, temperature=temperature)
