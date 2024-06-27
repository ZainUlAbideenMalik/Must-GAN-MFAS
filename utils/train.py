import torch
from tqdm import tqdm

def train_model(model, dataloader, criterion, gen_optimizer, disc_optimizer, num_epochs=300, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            gen_optimizer.zero_grad()
            disc_optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            gen_optimizer.step()
            disc_optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.4f}')

