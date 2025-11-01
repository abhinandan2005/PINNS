import torch
import matplotlib.pyplot as plt
import os

def predict(model, x, t):
    #Predict rho and u from model
    if x.ndim == 1:
        x = x.unsqueeze(1)
    if t.ndim == 1:
        t = t.unsqueeze(1)

    device = next(model.parameters()).device
    x, t = x.to(device), t.to(device)

    rho, u = model(x, t)
    return rho.detach().cpu(), u.detach().cpu()

def save_model(model, path):
    # a trained PyTorch model
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f" Model saved at: {path}")

def plot_solution(pinn_L, pinn_R, t_fixed=0.5, save_path=None):
    #Plot both rho and u at a fixed time
    device = next(pinn_L.parameters()).device

    # Generate spatial domain
    xL = torch.linspace(-1, 0, 200)
    xR = torch.linspace(0, 1, 200)
    t = torch.full_like(xL, t_fixed)

    # Ensure column shapes
    if xL.ndim == 1: xL = xL.unsqueeze(1)
    if xR.ndim == 1: xR = xR.unsqueeze(1)
    if t.ndim == 1:  t = t.unsqueeze(1)

    # Predict
    rho_L, u_L = predict(pinn_L, xL, t)
    rho_R, u_R = predict(pinn_R, xR, t)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot rho
    ax1.plot(xL.cpu(), rho_L, label="Left Subdomain", color="blue", linewidth=2)
    ax1.plot(xR.cpu(), rho_R, label="Right Subdomain", color="red", linewidth=2)
    ax1.axvline(0, color="k", linestyle="--", linewidth=1)
    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel(r"$\rho(x, t)$", fontsize=12)
    ax1.legend()
    ax1.set_title(f"Density ρ at t = {t_fixed}", fontsize=13)
    ax1.grid(True, alpha=0.3)

    # Plot u
    ax2.plot(xL.cpu(), u_L, label="Left Subdomain", color="blue", linewidth=2)
    ax2.plot(xR.cpu(), u_R, label="Right Subdomain", color="red", linewidth=2)
    ax2.axvline(0, color="k", linestyle="--", linewidth=1)
    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel(r"$u(x, t)$", fontsize=12)
    ax2.legend()
    ax2.set_title(f"Velocity u at t = {t_fixed}", fontsize=13)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved at: {save_path}")
    
    plt.show()

def plot_spacetime(pinn_L, pinn_R, save_path=None):
    #Create a spacetime heatmap of the solution
    # Create mesh
    x = torch.linspace(-1, 1, 300)
    t = torch.linspace(0, 1, 300)
    X, T = torch.meshgrid(x, t, indexing='ij')
    
    x_flat = X.flatten().unsqueeze(1)
    t_flat = T.flatten().unsqueeze(1)
    
    # Split into left and right
    left_mask = x_flat <= 0
    right_mask = x_flat > 0
    
    # Predict
    rho_L, u_L = predict(pinn_L, x_flat[left_mask.squeeze()], t_flat[left_mask.squeeze()])
    rho_R, u_R = predict(pinn_R, x_flat[right_mask.squeeze()], t_flat[right_mask.squeeze()])
    
    # Combine results
    rho_full = torch.zeros_like(x_flat)
    u_full = torch.zeros_like(x_flat)
    
    rho_full[torch.squeeze(left_mask)] = rho_L
    rho_full[torch.squeeze(right_mask)] = rho_R
    u_full[torch.squeeze(left_mask)] = u_L
    u_full[torch.squeeze(right_mask)] = u_R
    
    rho_grid = rho_full.reshape(300, 300)
    u_grid = u_full.reshape(300, 300)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    im1 = ax1.contourf(X.numpy(), T.numpy(), rho_grid.numpy(), levels=50, cmap='viridis')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('t', fontsize=12)
    ax1.set_title(r'Density $\rho(x,t)$', fontsize=13)
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.contourf(X.numpy(), T.numpy(), u_grid.numpy(), levels=50, cmap='plasma')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('t', fontsize=12)
    ax2.set_title(r'Velocity $u(x,t)$', fontsize=13)
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Spacetime plot saved at: {save_path}")
    
    plt.show()