import os
import torchvision 
from torchvision import transforms
from torch.utils.data import DataLoader, DistributedSampler
import pickle
from autoencoder_fra_vm import AE
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


with open("cropped_flipped_colored_dataset_01.txt", "rb") as file:
    data = pickle.load(file)
with open("cropped_flipped_colored_dataset_validation_01.txt", "rb") as file:
    validation = pickle.load(file)

validation =  validation[:2000]

training_data = []
validation_data = []

print("Loading train data")
for i in tqdm(range(len(data))):
    training_data.append(torch.tensor(data[i].T, dtype=torch.float))
print("Loading validation data")
for i in tqdm(range(len(validation))):
    validation_data.append(torch.tensor(validation[i].T, dtype=torch.float))


loader = torch.utils.data.DataLoader(training_data, batch_size=128)
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=128)
loader2 = torch.utils.data.DataLoader(training_data, batch_size=2, shuffle=True)
test = next(iter(loader2))
test2 = next(iter(loader2))
test3 = next(iter(loader2))
test4 = next(iter(loader2))

autoencoder = AE()
checkpoint_file = os.path.join('tmp', 'autoencoder_model_128_01')
#autoencoder.load_state_dict(torch.load(checkpoint_file))
train_loss = []
validation_loss = []
for i in tqdm(range(60)):
    loss = autoencoder.train(loader)
    print("Epoch: ", i, ", Train loss: ", loss)
    train_loss.append(loss)

    val_loss = autoencoder.test_epoch(validation_loader)
    print("Epoch: ", i, ", Validation loss: ", val_loss)
    validation_loss.append(val_loss)

torch.save(autoencoder.state_dict(), checkpoint_file)

plt.plot(train_loss, label="Train loss")
plt.plot(validation_loss, label="Validation loss")
plt.show()

res = autoencoder(test)
res2 = autoencoder(test2)
res3 = autoencoder(test3)
res4 = autoencoder(test4)

f, axarr = plt.subplots(4,2)
axarr[0][0].imshow(test.cpu().detach().numpy()[0].T)
axarr[0][1].imshow(res.cpu().detach().numpy()[0].T)
axarr[1][0].imshow(test2.cpu().detach().numpy()[0].T)
axarr[1][1].imshow(res2.cpu().detach().numpy()[0].T)
axarr[2][0].imshow(test3.cpu().detach().numpy()[0].T)
axarr[2][1].imshow(res3.cpu().detach().numpy()[0].T)
axarr[3][0].imshow(test4.cpu().detach().numpy()[0].T)
axarr[3][1].imshow(res4.cpu().detach().numpy()[0].T)
plt.show()
