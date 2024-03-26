# IMPORTS
import os
from autoencoder import *
import wandb
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from data_preprocessing import load_data, preprocess_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ['WANDB_API_KEY']="..."

# LOAD SENTENCES
text, labels = load_data()
inputs = preprocess_data(text, tokenizer)

# TRAIN MODEL
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_2.parameters(), lr=0.001, momentum=0.9)
batch_size = 128
train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs, labels, test_size=0.2, random_state=42)
train_dataset = TensorDataset(train_inputs, train_labels)
test_dataset = TensorDataset(test_inputs, test_labels)
print("Training Data: ",train_inputs.shape, train_labels.shape)
print("Test Data: ",test_inputs.shape, test_labels.shape)

wandb.init(project="...")
config = wandb.config
config.epochs = 3 
config.batch_size = batch_size

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

for epoch in range(config.epochs): 
    wandb.log({"epoch":epoch})

    loss_per_epoch = []
    accuracy_per_epoch = []
    correct = 0
    epoch_loss = 0
    running_loss = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        input_ids = inputs.to(device)

        optimizer.zero_grad()
        m,v,e = model_2(input_ids)
        outputs = (m,v)
        w , labels_ext = sample(outputs, labels)
        predictions = torch.argmax(w, dim = 1)

        loss = criterion(w, labels_ext)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_loss += (loss.item() * batch_size)
        correct += (predictions == labels_ext).sum().item()

        if i%10 == 0:
            running_loss /= 10
            print("Running Loss ",i,"-th iteration: ", running_loss)
            wandb.log({"running_loss": running_loss})
            running_loss = 0

    epoch_loss /= (train_inputs.shape[0])
    loss_per_epoch.append(epoch_loss)
    accuracy_per_epoch.append(correct/(train_inputs.shape[0]*100))

    wandb.log({"epoch_loss": epoch_loss})
    wandb.log({"epoch_accuracy": correct/(train_inputs.shape[0])})

print('Finished Training')
torch.save(model_2.state_dict(), './Saved_Models/autoencoderv4.pth')

# Testing:

test_loss = 0
correct = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        input_ids = inputs.to(device)
        m,v,e = model_2(input_ids)
        outputs = (m,v)
        w , labels_ext = sample(outputs, labels)
        predictions = torch.argmax(w, dim = 1)
        loss = criterion(w, labels_ext)

        test_loss += loss.item() * batch_size
        correct += (predictions == labels_ext).sum().item()
print("Test Accuracy: ",correct/test_inputs.shape[0])
print("Test Loss: ",test_loss/test_inputs.shape[0])

wandb.finish()