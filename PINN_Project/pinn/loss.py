# Loss function with correct physics
import torch
from pinn.physics import pde_residual

def loss_fn(pinn_L, pinn_R, xL, tL, xR, tR, x_ic, t_ic, rho_ic, u_ic, 
            x_i, t_i, x_left, x_right, t_b):
    """
    Total loss = PDE loss + IC loss + Interface continuity + Boundary conditions
    """
    
    # 1. PDE residuals
    res_rho_L, res_u_L = pde_residual(pinn_L, xL, tL)
    res_rho_R, res_u_R = pde_residual(pinn_R, xR, tR)
    
    L_pde = (torch.mean(res_rho_L**2) + torch.mean(res_u_L**2) + 
             torch.mean(res_rho_R**2) + torch.mean(res_u_R**2))
    
    # 2. Initial condition loss
    rho_pred_L, u_pred_L = pinn_L(x_ic, t_ic)
    rho_pred_R, u_pred_R = pinn_R(x_ic, t_ic)
    
    L_ic = (torch.mean((rho_pred_L - rho_ic)**2) + torch.mean((u_pred_L - u_ic)**2) +
            torch.mean((rho_pred_R - rho_ic)**2) + torch.mean((u_pred_R - u_ic)**2))
    
    # 3. Interface continuity (both rho and u must be continuous)
    rho_L_i, u_L_i = pinn_L(x_i, t_i)
    rho_R_i, u_R_i = pinn_R(x_i, t_i)
    
    L_interface = torch.mean((rho_L_i - rho_R_i)**2) + torch.mean((u_L_i - u_R_i)**2)
    
    # 4. Zero-gradient boundary conditions at x = -1 and x = 1
    # ∂rho/∂x = 0 and ∂u/∂x = 0 at boundaries
    x_left.requires_grad = True
    x_right.requires_grad = True
    
    rho_left, u_left = pinn_L(x_left, t_b)
    rho_right, u_right = pinn_R(x_right, t_b)
    
    # Compute spatial gradients
    rho_x_left = torch.autograd.grad(rho_left, x_left, 
                                      grad_outputs=torch.ones_like(rho_left),
                                      create_graph=True)[0]
    u_x_left = torch.autograd.grad(u_left, x_left,
                                    grad_outputs=torch.ones_like(u_left),
                                    create_graph=True)[0]
    
    rho_x_right = torch.autograd.grad(rho_right, x_right,
                                       grad_outputs=torch.ones_like(rho_right),
                                       create_graph=True)[0]
    u_x_right = torch.autograd.grad(u_right, x_right,
                                     grad_outputs=torch.ones_like(u_right),
                                     create_graph=True)[0]
    
    L_bc = (torch.mean(rho_x_left**2) + torch.mean(u_x_left**2) +
            torch.mean(rho_x_right**2) + torch.mean(u_x_right**2))
    
    # Total loss with weights
    total_loss = L_pde + L_ic + 10.0 * L_interface + L_bc
    
    return total_loss, (L_pde.item(), L_ic.item(), L_interface.item(), L_bc.item())