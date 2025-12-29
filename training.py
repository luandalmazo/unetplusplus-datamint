from datamint import Api 
from TMJDataset import TMJDataset2D   
from torch.utils.data import DataLoader
from trainerTMJ import UNetPPModule
import torch
from datamint.mlflow.lightning.callbacks import MLFlowModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import EarlyStopping
from datamint.mlflow import set_project
import lightning as L

if __name__ == "__main__":    
    PROJECT_NAME = "TMJ Study"
    IMAGE_SIZE = 256
    NUM_CLASSES = 4

    api = Api()
    proj = api.projects.get_by_name(PROJECT_NAME)

    if proj is None:
        raise ValueError(f"Project '{PROJECT_NAME}' does not exist.")
    else:
        print(f"Project '{PROJECT_NAME}' found.")


    all_resources = list(api.resources.get_list(project_name=PROJECT_NAME))
    all_resources.sort(key=lambda r: r.filename)
    
    all_resources = all_resources[:10]

    ''' Split the resources into training, validation, and test sets'''
    n_total = len(all_resources)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val
    train_resources = all_resources[:n_train]
    val_resources = all_resources[n_train:n_train + n_val]
    test_resources = all_resources[n_train + n_val:]
    
    print("Resources splitted...")
    
    api.resources.add_tags(train_resources, ['split:train'])
    api.resources.add_tags(val_resources, ['split:val'])
    api.resources.add_tags(test_resources, ['split:test'])

    print(f"Total resources: {n_total}")
    print(f"Training: {len(train_resources)}")
    print(f"Validation: {len(val_resources)}")
    print(f"Test: {len(test_resources)}")
    
    BATCH_SIZE = 16 
    NUM_WORKERS = 4 

    ''' Create datasets '''
    print("Building training dataset...")
    train_dataset = TMJDataset2D(
        split='train',
    )
    print(f"  Training samples (slices): {len(train_dataset)}")

    print("Building validation dataset...")
    val_dataset = TMJDataset2D(
        split='val',
    )
    print(f"  Validation samples (slices): {len(val_dataset)}")

    print("Building test dataset...")
    test_dataset = TMJDataset2D(
        split='test',
    )
    print(f"  Test samples (slices): {len(test_dataset)}")

    ''' Create dataloaders '''
    print("Creating dataloaders...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    print("Dataloaders created.")
    print("Starting model setup...")
    print(f"Number of classes: {NUM_CLASSES}, Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    
    model = UNetPPModule(
    num_classes=NUM_CLASSES,
    encoder_name='resnet34',  
    learning_rate=1e-4,
    )

    print("Model initialized.")
    print("Testing model forward pass and loss computation...")
    
    with torch.no_grad():
        sample_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        sample_output = model(sample_input)
        print(f"Input shape: {sample_input.shape}")
        print(f"Output shape: {sample_output.shape}")

        # test loss computation
        sample_target = train_dataloader.dataset[-1]['mask']  # (1, H, W)
        sample_loss = model.criterion(sample_output, sample_target.unsqueeze(0))
        print(f"Sample loss: {sample_loss.item():.4f}")

    print("It seems everything is set up correctly.")
    
    set_project(PROJECT_NAME)

    print("Configuring training callbacks...")
    
    ''' Model checkpoint callback'''
    checkpoint_callback = MLFlowModelCheckpoint(
        monitor="val/iou",                    # Metric to monitor
        mode="max",                           # Save when metric increases
        save_top_k=1,                         # Keep only the best model
        filename="best_unetpp",               # Checkpoint filename
        save_weights_only=True,              # Save full model state
        register_model_name=PROJECT_NAME,     # Name in Model Registry
        register_model_on='test',             # Register after test evaluation
    )

    ''' Early stopping callback '''
    early_stop_callback = EarlyStopping(
        monitor="val/iou",
        mode="max",
        patience=10                          # Stop if no improvement for 10 epochs
    )

    ''' MLflow logger '''
    mlflow_logger = MLFlowLogger(
        experiment_name=f"{PROJECT_NAME}_training",
        run_name="unetpp_resnet34_busi",
    )

    print("Training callbacks configured.")    
    print("Setting up trainer...")
    trainer = L.Trainer(
        max_epochs=50,                        # Maximum training epochs
        logger=mlflow_logger,                 # MLflow logging
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator='auto',                   # Auto-detect GPU/CPU
        devices=1,                            # Single device
        precision='16-mixed',                 # Mixed precision for faster training
        log_every_n_steps=10,                 # Log frequency
        num_sanity_val_steps=2,               # Validation sanity check
    )

    print("🚀 Starting training...")
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
    print("Opening project in DataMint... (Browser will open)")
    proj.show()
    
    print("🔍 Evaluating on test set...")
    test_results = trainer.test(dataloaders=test_dataloader)

    print("\n✅ Training complete!")
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best validation IoU: {checkpoint_callback.best_model_score:.4f}")





