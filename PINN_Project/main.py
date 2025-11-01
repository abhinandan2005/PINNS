import torch 
from pinn import (
    MLP,
    generate_collocation_points,
    initial_condition,
    interface_points,
    boundary_points,
    train,
    utils
)

if __name__ == "__main__":
    # Setting up the device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize two subdomain models (to account for the discontinuity)
    pinn_L = MLP().to(device)
    pinn_R = MLP().to(device)
    
    # Generate training data
    (xL, tL), (xR, tR) = generate_collocation_points(N_f=2000)
    x_ic, t_ic, rho_ic, u_ic = initial_condition(N_ic=500)
    x_i, t_i = interface_points(N_i=200)
    x_left, x_right, t_b = boundary_points(N_b=100)

    # Move tensors to device
    tensors = [xL, tL, xR, tR, x_ic, t_ic, rho_ic, u_ic, x_i, t_i, x_left, x_right, t_b]
    xL, tL, xR, tR, x_ic, t_ic, rho_ic, u_ic, x_i, t_i, x_left, x_right, t_b = [
        arr.to(device) for arr in tensors
    ]

    # Package data
    data = ((xL, tL), (xR, tR), (x_ic, t_ic, rho_ic, u_ic), (x_i, t_i), (x_left, x_right, t_b))

    # Train XPINN
    train(pinn_L, pinn_R, data, epochs=2000, lr=1e-3, batch_size=256)

    # Save trained models
    utils.save_model(pinn_L, "PINN_Project/checkpoints/pinn_L.pth")
    utils.save_model(pinn_R, "PINN_Project/checkpoints/pinn_R.pth")

    # Plotting our results
    utils.plot_spacetime(pinn_L, pinn_R, 
                        save_path="PINN_Project/results/spacetime.png")
    
    utils.plot_solution(pinn_L, pinn_R, t_fixed=0, 
                       save_path="PINN_Project/results/solution_t00.png")
    
    utils.plot_solution(pinn_L, pinn_R, t_fixed=0.5, 
                       save_path="PINN_Project/results/solution_t05.png")
    
    utils.plot_solution(pinn_L, pinn_R, t_fixed=1, 
                       save_path="PINN_Project/results/solution_t10.png")