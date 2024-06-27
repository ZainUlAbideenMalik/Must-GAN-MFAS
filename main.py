import os
import torch
import torch.optim as optim
import torch.nn as nn
import logging
from dataloading import get_casia_surf_cefa_dataloader, get_casia_surf_dataloader, get_wmca_dataloader
from models import model
from utils import train_model, evaluate_model, save_checkpoint, load_checkpoint

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    train_data_dir_cefa = 'D/FAS/casia-surf-cefa'
    train_data_dir_surf = 'D/FAS/casia-surf'
    eval_data_dir = 'D/FAS/wmca'

    batch_size = 16
    num_epochs = 300
    checkpoint_path = 'checkpoint.pth.tar'

    logging.info("Loading datasets...")

    train_loader_cefa = get_casia_surf_cefa_dataloader(train_data_dir_cefa, batch_size=batch_size,shuffle=True, is_train=True)
    train_loader_surf = get_casia_surf_dataloader(train_data_dir_surf, batch_size=batch_size,shuffle=True, is_train=True)
    eval_loader = get_wmca_dataloader(eval_data_dir, batch_size=batch_size,shuffle=False, is_train=False)
    

    logging.info("Initializing model and optimizer...")
    model = model(num_modalities=3)  
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.MSELoss()

    if os.path.isfile(checkpoint_path):
        logging.info("Loading checkpoint...")
        load_checkpoint(model, optimizer, filename=checkpoint_path)

    logging.info("Starting training...")
    for epoch in range(num_epochs):
        logging.info(f'Training on CASIA-Surf CeFA dataset...')
        train_loss_cefa = train_model(model, train_loader_cefa, criterion, optimizer, num_epochs=1)
        
        logging.info(f'Epoch {epoch + 1}/{num_epochs}, CASIA-Surf CeFA Training Loss: {train_loss_cefa:.4f}')

        logging.info(f'Training on CASIA-Surf dataset...')
        train_loss_surf = train_model(model, train_loader_surf, criterion, optimizer, num_epochs=1)

        logging.info(f'Epoch {epoch + 1}/{num_epochs}, CASIA-Surf Training Loss: {train_loss_surf:.4f}')

        is_best = epoch == num_epochs - 1
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best, filename=checkpoint_path)

        if (epoch + 1) % 5 == 0:
            logging.info("Evaluating model on WMCA dataset...")
            eval_loss, apcer, bpcer, acer, eer, tpr_at_fpr, hter = evaluate_model(model, eval_loader, criterion)

            logging.info(f'Epoch {epoch + 1}/{num_epochs}, WMCA Evaluation Loss: {eval_loss:.4f}')
            logging.info(f'APCER: {apcer:.4f}, BPCER: {bpcer:.4f}, ACER: {acer:.4f}, EER: {eer:.4f}, TPR@FPR=10^-4: {tpr_at_fpr:.4f}, HTER: {hter:.4f}')

    logging.info("Training completed.")

if __name__ == "__main__":
    main()
