from train_standard_models import main
from configurations import configurations_standard_v2
import os

# Path configuration
home_path = '/home/nct/nct01150/oscar/standard/experiment11_standard_overfitting'


# Model names to test 
model_names = ['AlexNet', 'StandardArchitectureModel']

# Calculate total experiments
total_experiments = len(model_names) * len(configurations_standard_v2)

for model_name in model_names:
    for config in configurations_standard_v2:
        
        print(f"\n{'='*60}")
        print(f"Starting: {model_name} - Configuration: {config}")
        print(f"{'='*60}\n")
        
        # Create model-specific folder structure: results/ModelName/
        model_folder = os.path.join(home_path, "results", model_name)
        os.makedirs(model_folder, exist_ok=True)
        
        # Define output file path: results/models/ModelName/scenario.csv
        csv_filename = os.path.join(model_folder, f"{config['scenario']}.csv")

        # Call main with the specific parameters
        main(
            model_name=model_name,
            batch_size=config["batch_size"],
            epochs=config["epochs"],
            csv_filename=csv_filename,
            lr=config["learning_rate"],
            step_size=config["lr_step"],
            gamma=config["lr_decay"],
            betas=config["optimizer_betas"])

print(f"\n{'='*60}")
print("All experiments completed!")
print(f"{'='*60}\n")
