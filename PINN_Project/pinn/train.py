# Training loop
import torch
from pinn.loss import loss_fn

def train(pinn_L, pinn_R, data, epochs=5000, lr=1e-3, batch_size=256):
    """
    Training loop with batching support
    
    Args:
        batch_size: Number of samples per batch (default: 256)
    """
    from pinn.data import create_dataloaders
    
    # Create batched dataloaders
    loaders = create_dataloaders(data, batch_size=batch_size)
    
    optimizer = torch.optim.Adam(
        list(pinn_L.parameters()) + list(pinn_R.parameters()), 
        lr=lr
    ) 
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=500
    )

    for epoch in range(epochs+1):
        epoch_loss = 0.0
        epoch_components = {'pde': 0.0, 'ic': 0.0, 'interface': 0.0, 'bc': 0.0}
        n_batches = 0
        
        # Iterate through batches
        # We zip all dataloaders to process corresponding batches together
        for batch_data in zip(loaders['collocation_L'], 
                              loaders['collocation_R'],
                              loaders['initial'],
                              loaders['interface'],
                              loaders['boundary']):
            
            (xL, tL), (xR, tR), (x_ic, t_ic, rho_ic, u_ic), (x_i, t_i), (x_left, x_right, t_b) = batch_data
            
            optimizer.zero_grad()
            
            total_loss, (Lp, Li, Lint, Lbc) = loss_fn(
                pinn_L, pinn_R, xL, tL, xR, tR, 
                x_ic, t_ic, rho_ic, u_ic, 
                x_i, t_i, x_left, x_right, t_b
            )
            
            total_loss.backward()
            optimizer.step()
            
            # Accumulate losses for reporting
            epoch_loss += total_loss.item()
            epoch_components['pde'] += Lp
            epoch_components['ic'] += Li
            epoch_components['interface'] += Lint
            epoch_components['bc'] += Lbc
            n_batches += 1
        
        # Average losses over all batches
        avg_loss = epoch_loss / n_batches
        scheduler.step(avg_loss)
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Total={avg_loss:.4e}, "
                  f"PDE={epoch_components['pde']/n_batches:.2e}, "
                  f"IC={epoch_components['ic']/n_batches:.2e}, "
                  f"Interface={epoch_components['interface']/n_batches:.2e}, "
                  f"BC={epoch_components['bc']/n_batches:.2e}")

    print("Training done.")