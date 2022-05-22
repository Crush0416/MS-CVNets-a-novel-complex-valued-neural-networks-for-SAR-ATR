
""" 
       training on Multi-Stream Complex Value Network---MS-CVNets
"""

# import
from MSCVNets import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import gc

#---------------------__Main__-----------------------
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print ('GPU is true')
    print('cuda version: {}'.format(torch.version.cuda))
else:
    print('CPU is true')
    
#  hyperparameters        
batch_size = 32
num_classes = 10
num_epochs = 350
learning_rate = 5e-4
seed = 10086

# initial seeds
seeds_init(seed)

#----------------DataLoader-----------------
train_dataset = scipy.io.loadmat('./Complex_MSTAR/data_SOC/class10/train/2equilibrium/data_train_64.mat')
test_dataset  = scipy.io.loadmat('./Complex_MSTAR/data_SOC/class10/test/2equilibrium/data_test_64.mat')

traindata_r = train_dataset['data_r']
traindata_i = train_dataset['data_i']
trainlabel = train_dataset['label'].squeeze()    ##  label必须是一维向量

testdata_r = test_dataset['data_r']
testdata_i = test_dataset['data_i']
testlabel = test_dataset['label'].squeeze()

train_dataset = MyDataset(img_r=traindata_r, img_i=traindata_i, label=trainlabel, transform=transforms.ToTensor())
test_dataset  = MyDataset(img_r=testdata_r, img_i=testdata_i, label=testlabel, transform=transforms.ToTensor())
print('real train data size: {} \nimaginary train data size: {}' \
      .format(train_dataset.img_r.shape[0], train_dataset.img_i.shape[0]))
print('real test data size: {} \nimaginary test data size: {}' \
      .format(test_dataset.img_r.shape[0], test_dataset.img_i.shape[0]))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#---------------------model preparation------------------
model = MSCVNet(num_classes).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, \
             # verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.8, last_epoch = -1)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [50,300], gamma = 0.5, last_epoch = -1)
#-------------------------training-----------------------
train_loss = []
val_loss = []
val_acc = []
total_step = len(trainlabel) // batch_size
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total = 0
    correct = 0
    for batch_idx, (image_r, image_i, label) in enumerate(train_loader):        
        image_r = image_r.to(device)
        image_i = image_i.to(device)
        label   = label.to(device)
        
        optimizer.zero_grad()
        output = model(image_r, image_i)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()
        if (batch_idx+1) % 20 == 0:
            print ('LR={}, Epoch [{}/{}], Step [{}/{}], Step Loss: {:.8f},  Total Loss: {:.8f}' 
                   .format(optimizer.param_groups[0]['lr'], epoch+1, num_epochs, batch_idx+1, total_step, loss.item(), total_loss))                     
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
    print('---------------------------training-----------------------------')    
    print('correct number : {}, test data number : {}, Accuracy : {}'.format(correct, total, 100 * correct / total))   
    train_loss.append(total_loss)
    #----------------Validation----------------
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        temp_loss = 0        
        label = []
        label_pre = []
        for image_r, image_i, target in test_loader:
            image_r = image_r.to(device)
            image_i = image_i.to(device)
            target  = target.to(device)
            output = model(image_r, image_i)
            loss = criterion(output, target)
            temp_loss += loss.detach().item()

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            label.append(target)
            label_pre.append(predicted)
            # label.extend(target.data.cpu().numpy())      # data form GPU to CPU
            # label_pre.extend(predicted.data.cpu().numpy())
        print('---------------------------validation---------------------------')  
        print('correct number : {}, test data number : {}, Accuracy : {}'.format(correct, total, 100 * correct / total))
        print('----------------------------------------------------------------')
        print('Training Error: {},  Valdation Error: {}'.format(total_loss, temp_loss))
        print('----------------------------------------------------------------')
        val_loss.append(temp_loss)
        val_acc.append(correct/total)
    #  save model
    if (correct/total) > 0.997:
        acc = ('%.4f'%(correct/total))
        savepath = './models/fullmodel_'+str(epoch+1)+'Ep_'+acc+'Acc.pth'
        torch.save(model,savepath)
    # scheduler.step()     ## 自适应动态调整学习率
    gc.collect()     #  清除缓存
val_acc_max, idx = torch.max(torch.Tensor(val_acc), -1)
val_loss1 = torch.Tensor(val_loss)[idx]
print('MS-CVNet64-Full: val_acc: {}, val_loss: {}, idx: {}'.format(val_acc_max, val_loss1, idx+1))           
#-----------trian loss curve--------------
plt.figure#(figsize=(10,5.625))
plt.title('train and val loss decay curves on MS-CVNet64-Full', fontsize=15)
plt.xlabel('Epochs', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.plot(train_loss, label='train_loss')
plt.plot(val_loss, label='val_loss')
plt.tick_params(labelsize=10)   #调整坐标轴刻度的字体大小
plt.legend(fontsize=10)       #参数调整train-loss与val-loss字体的大小
# plt.savefig("./pan1.jpg")
plt.show()
#---------------save model----------------
# torch.save(model,'./models/fullmodel_100Ep_1e-3lr.pth')
    
    


















 