from train_non_standard_models import main
from configurations import configurations_non_standard, configurations_non_standard_vit
import os

home_path = '/home/nct/nct01150/oscar/non-standard/experiment17'
model_names = ['InceptionV3', 'ViTModel']

# Calculate total experiments
total_experiments = len(model_names) * len(configurations_non_standard)

for model_name in model_names:
    if model_name == 'InceptionV3':
        print("\n\n======================== InceptionV3 Experiments ========================\n\n")
        for config in configurations_non_standard:
            
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
                momentum=config["momentum"]
                )
    elif model_name == 'ViTModel':
        print("\n\n======================== ViTModel Experiments ========================\n\n")
        for config in configurations_non_standard_vit:
            
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
                momentum=None
                )
print(f"\n{'='*60}")
print("All experiments completed!")
print(f"{'='*60}\n")
