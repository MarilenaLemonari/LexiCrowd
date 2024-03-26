# IMPORTS
from variationalmodel import *
import os
import wandb
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

os.environ['WANDB_API_KEY']="..."

# LOAD SENTENCES
text, labels = load_data()
inputs = preprocess_data(text, tokenizer)

# TRAIN MODEL:
batch_size = 128
train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs, labels, test_size=0.1, random_state=42)
train_dataset = TensorDataset(train_inputs, train_labels)
test_dataset = TensorDataset(test_inputs, test_labels)
print("Training Data: ",train_inputs.shape, train_labels.shape)
print("Test Data: ",test_inputs.shape, test_labels.shape)

optimizer = optim.Adam(model_3.parameters(), lr=0.001)
criterion = nn.MSELoss()
label_criterion = nn.CrossEntropyLoss()
emb_criterion = nn.CosineSimilarity(dim=1) 
def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = torch.norm(emb_criterion(x_hat, x))
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss, KLD


wandb.init(project="...")
config = wandb.config
config.epochs = 4
config.batch_size = batch_size

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

mode = "Embeddings" 
for epoch in range(config.epochs): 
    wandb.log({"epoch":epoch})

    total_loss_per_epoch = []
    recon_loss_per_epoch = []
    kld_loss_per_epoch = []
    mean_loss_per_epoch = []
    var_loss_per_epoch = []
    label_loss_per_epoch = []

    total_epoch_loss = 0
    recon_epoch_loss = 0
    kld_epoch_loss = 0
    mean_epoch_loss = 0
    var_epoch_loss = 0
    label_epoch_loss = 0

    running_loss = 0
    recon_running_loss = 0
    kld_running_loss = 0
    mean_running_loss = 0
    var_running_loss = 0
    label_running_loss = 0
    correct = 0
    

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        input_ids = inputs.to(device)

        optimizer.zero_grad()

        if mode == "Embeddings":
          [gt_mean, gt_var], w, [embeds_gt, embeds] = model_3(input_ids,"Embeddings")
          recon_loss , kld = loss_function(embeds_gt, embeds, gt_mean, gt_var)
          label_loss = label_criterion(w, labels.long())
          predictions = torch.argmax(w, dim = 1)
          label_loss_mean = label_criterion(gt_mean, labels.long())
          loss = 0.5 * recon_loss + kld + ( label_loss + label_loss_mean ) * 200
          loss_mean = 0
          loss_var = 0
        elif mode == "Ids":
          [gt_mean, gt_var], [mean, var] ,w, [embeds_gt, embeds] = model_3(input_ids,"Ids")
          recon_loss , kld = loss_function(embeds_gt, embeds, gt_mean, gt_var)
          label_loss_mean = label_criterion(mean, labels.long())
          label_loss_mean_2 = label_criterion(gt_mean, labels.long())
          label_loss = label_loss_mean + label_loss_mean_2
          predictions = torch.argmax(w, dim = 1)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_epoch_loss += (loss.item() * config.batch_size)

        recon_running_loss += recon_loss.item()
        recon_epoch_loss += (recon_loss.item() * config.batch_size)
        kld_running_loss += kld.item()
        kld_epoch_loss += (kld.item() * config.batch_size)

        label_running_loss += label_loss.item()
        label_epoch_loss += (label_loss.item() * config.batch_size)
        correct += ((predictions == labels.long()).sum().item()*(100/config.batch_size))

        if i%10 == 0 and i!=0:
            running_loss /= 10
            recon_running_loss /= 10
            kld_running_loss /= 10
            #
            mean_running_loss /= 10
            var_running_loss /= 10

            label_running_loss /= 10
            correct /= 10
            print("Running Loss ",i,"-th iteration: ", running_loss)
            print("Embeddings running loss: ",recon_running_loss, ", KLD: ", kld_running_loss)
            print("Label running loss: ",label_running_loss, "Label accuracy: ", correct)
            wandb.log({"running_loss": running_loss, "embeds_run_loss": recon_running_loss, "kld_run": kld_running_loss})
            wandb.log({"label_run_loss": label_running_loss, "label_run_accuracy": correct})
            running_loss = 0
            recon_running_loss = 0
            kld_running_loss = 0
            mean_running_loss = 0
            var_running_loss = 0
            label_running_loss = 0
            correct = 0


    total_epoch_loss /= (train_inputs.shape[0])
    recon_epoch_loss /= (train_inputs.shape[0])
    kld_epoch_loss /= (train_inputs.shape[0])
    mean_epoch_loss /= (train_inputs.shape[0])
    var_epoch_loss /= (train_inputs.shape[0])
    label_epoch_loss /= (train_inputs.shape[0])
    
    total_loss_per_epoch.append(total_epoch_loss)
    recon_loss_per_epoch.append(recon_epoch_loss)
    kld_loss_per_epoch.append(kld_epoch_loss)
    mean_loss_per_epoch.append(mean_epoch_loss)
    var_loss_per_epoch.append(var_epoch_loss)
    label_loss_per_epoch.append(label_epoch_loss)
    
    wandb.log({"total_epoch_loss": total_epoch_loss, "embeds_epoch_loss": recon_epoch_loss, "kld_epoch_loss": kld_epoch_loss})
    wandb.log({"label_epoch_loss": label_epoch_loss})
    
print('Finished Training')
torch.save(model_3.state_dict(), './Saved_Models/VAEv1.pth')
      
# Testing:
test_total_losses = []
test_recon_losses = []
test_kld_losses = []
test_mean_losses = []
test_var_losses = []
test_label_losses = []

test_loss = 0
test_recon_loss = 0
test_kld_loss = 0
test_mean_loss = 0
test_var_loss = 0
test_label_loss = 0
correct = 0
with torch.no_grad():
    for i, data in tqdm(enumerate(test_loader, 0)):
        inputs, labels = data
        input_ids = inputs.to(device)

        optimizer.zero_grad()

        if mode == "Embeddings":
          [gt_mean, gt_var], w, [embeds_gt, embeds] = model_3(input_ids,"Embeddings")
          recon_loss , kld = loss_function(embeds_gt, embeds, gt_mean, gt_var)
          label_loss = label_criterion(w, labels.long())
          predictions = torch.argmax(w, dim = 1)
          label_loss_mean = label_criterion(gt_mean, labels.long())
          loss = recon_loss + kld + label_loss + label_loss_mean 
          loss_mean = 0
          loss_var = 0
        elif mode == "Ids":
          [gt_mean, gt_var], [mean, var] ,w, [embeds_gt, embeds] = model_3(input_ids,"Ids")
          recon_loss , kld = loss_function(embeds_gt, embeds, gt_mean, gt_var)
          label_loss = label_criterion(w, labels)
          predictions = torch.argmax(w, dim = 1)


        test_loss += (loss.item() * batch_size)
        test_recon_loss += (recon_loss.item() * batch_size)
        test_kld_loss += (kld.item() * batch_size)
        test_label_loss += (label_loss_mean.item() * batch_size)
        correct +=  (predictions == labels.long()).sum().item() 


    test_loss /= (test_inputs.shape[0])
    test_recon_loss /= (test_inputs.shape[0])
    test_kld_loss /= (test_inputs.shape[0])
    test_mean_loss /= (test_inputs.shape[0])
    test_var_loss /= (test_inputs.shape[0])
    test_label_loss /= (test_inputs.shape[0])
    correct *= (100/test_inputs.shape[0]) 
    
    test_total_losses.append(test_loss)
    test_recon_losses.append(test_recon_loss)
    test_kld_losses.append(test_kld_loss)
    test_mean_losses.append(test_mean_loss)
    test_var_losses.append(test_var_loss)
    test_label_losses.append(test_label_loss)

print("Total test losses:", test_total_losses)
print("Embeddings test losses:",test_recon_losses)
print("KLD:",test_kld_losses)
print("Label test losses:",test_label_losses)
print("Label test accuracy:", correct)

wandb.finish()