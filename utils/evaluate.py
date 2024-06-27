import torch
from metrics import calculate_apcer, calculate_bpcer, calculate_acer, calculate_eer, calculate_tpr_at_fpr, calculate_hter

def evaluate_model(model, dataloader, criterion, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.to(device)
    model.eval()
    running_loss = 0.0
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            running_loss += loss.item() * inputs.size(0)

            all_scores.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    total_loss = running_loss / len(dataloader.dataset)

    scores = np.array(all_scores)

    labels = np.array(all_labels)

    threshold = 0.5  
    apcer = calculate_apcer(scores, labels, threshold)

    bpcer = calculate_bpcer(scores, labels, threshold)

    acer = calculate_acer(apcer, bpcer)

    eer = calculate_eer(scores, labels)

    tpr_at_fpr = calculate_tpr_at_fpr(scores, labels)
    
    hter = calculate_hter(apcer, bpcer)

    print(f'Evaluation Loss: {total_loss:.4f}')
    print(f'APCER: {apcer:.4f}, BPCER: {bpcer:.4f}, ACER: {acer:.4f}, EER: {eer:.4f}, TPR@FPR=10^-4: {tpr_at_fpr:.4f}, HTER: {hter:.4f}')

    return total_loss, apcer, bpcer, acer, eer, tpr_at_fpr, hter
