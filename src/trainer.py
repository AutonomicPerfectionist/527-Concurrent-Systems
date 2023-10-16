import torch.optim as optim
import torch.nn as nn
import torch

scaler = torch.cuda.amp.GradScaler()

def training_step(model, trainloader, epoch, device, quantized=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # Get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to('cuda:1')
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        if quantized:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # Print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    print('Epoch ', epoch, 'finished training')
