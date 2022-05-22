'''
     testing on MS-CVNet
'''

from MSCVNets import *
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#-----------------DEVICE CONFIGURATION--------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print ('GPU is true')
    print('cuda version: {}'.format(torch.version.cuda))
else:
    print('CPU is true')
    
#  hyperparameters        
PATH = './models/class10/fullmodel_434Ep_0.9979Acc.pth'      ## model path
batch_size = 100
num_classes = 10

#----------------DataLoader-----------------
train_dataset = scipy.io.loadmat('./Complex_MSTAR/data_SOC/class10/train/2equilibrium/data_train_64.mat')
test_dataset = scipy.io.loadmat('./Complex_MSTAR/data_SOC/class10/test/2equilibrium/data_test_64.mat')

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

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#----------------model loading-----------------
model = torch.load(PATH)
# criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()

#----------------testing-----------------
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    total_loss = 0
    label = []
    label_pre = []
    for image_r, image_i, target in train_loader:
        image_r = image_r.to(device)
        image_i = image_i.to(device)
        target  = target.to(device)
        output = model(image_r, image_i)  
        loss = criterion(output, target)
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)      
        total += target.size(0)
        correct += (predicted == target).sum().item()           
        label.extend(target.data.cpu().numpy())
        label_pre.extend(predicted.data.cpu().numpy())
    print('correct number : {}, train data number : {}, train Accuracy: {}, trian_loss: {:.4f}'.format(correct, total, 100*correct/total, total_loss))
    matrix = confusion_matrix(label,label_pre)
    print('############ confusion matrix ########### \n', matrix)
    # scipy.io.savemat('./results/confusion_matrix.mat',{'confusion_matrix':matrix,'label':label, 'label_predict':label_pre})

with torch.no_grad():
    correct = 0
    total = 0
    total_loss = 0
    label = []
    label_pre = []
    for image_r, image_i, target in test_loader:
        image_r = image_r.to(device)
        image_i = image_i.to(device)
        target  = target.to(device)
        output = model(image_r, image_i) 
        loss = criterion(output, target)
        total_loss += loss.item()        
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()           
        label.extend(target.data.cpu().numpy())
        label_pre.extend(predicted.data.cpu().numpy())
    print('correct number : {}, test data number : {}, test Accuracy: {}, test loss: {:.4f}'.format(correct, total, 100*correct/total, total_loss))    
    # label = np.asarray(label)
    # label_pre = np.asarray(label_pre)
    matrix = confusion_matrix(label,label_pre)
    print('############ confusion matrix ########### \n', matrix)
    # scipy.io.savemat('./results/confusion_matrix_class3.mat',{'confusion_matrix':matrix,'label':label, 'label_predict':label_pre})






























