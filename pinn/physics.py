# Computes the PDE residuals for the conservation law system
import torch

def pde_residual(model, x, t):
    """
    Compute residuals for:
    ∂ρ/∂t + ∂u/∂x = 0 
    ∂u/∂t + ∂ρ/∂x = 0  
    """
    x.requires_grad = True
    t.requires_grad = True
    
    rho, u = model(x, t)
    
    # Compute time derivatives
    rho_t = torch.autograd.grad(rho, t, grad_outputs=torch.ones_like(rho),
                                create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]
    
    # Compute spatial derivatives
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]
    rho_x = torch.autograd.grad(rho, x, grad_outputs=torch.ones_like(rho),
                                create_graph=True)[0]
    
    # PDE residuals
    residual_1 = rho_t + u_x  # ∂ρ/∂t + ∂u/∂x = 0
    residual_2 = u_t + rho_x  # ∂u/∂t + ∂ρ/∂x = 0
    
    return residual_1, residual_2