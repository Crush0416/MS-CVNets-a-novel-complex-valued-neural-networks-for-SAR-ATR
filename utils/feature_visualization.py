'''
     testing on MS-CVNet
'''

from MSCVNets import *
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

#-----------------DEVICE CONFIGURATION--------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print ('GPU is true')
    print('cuda version: {}'.format(torch.version.cuda))
else:
    print('CPU is true')
    
#  hyperparameters        
PATH = './models/class10/fullmodel_434Ep_0.9979Acc.pth'      ## model path

#----------------DataLoader-----------------
train_dataset1 = scipy.io.loadmat('../../../Datasets/MSTAR_ZZQ/data_SOC/class10/2train/data_train_64.mat')
train_dataset2 = scipy.io.loadmat('../../../Datasets/Complex_MSTAR/data_SOC/class10/train/2equilibrium/data_train_64.mat')


traindata_am = train_dataset1['data_am']
traindata_ph = train_dataset1['data_ph']
label1 = train_dataset1['label'].squeeze()    ##  label必须是一维向量

traindata_r = train_dataset2['data_r']
traindata_i = train_dataset2['data_i']
label2 = train_dataset2['label'].squeeze()

image_am, image_ph = traindata_am[1086,0,:,:].squeeze(), traindata_ph[1086,0,:,:].squeeze()
image_r, image_i   = traindata_r[1086,0,:,:].squeeze(), traindata_i[1086,0,:,:].squeeze()

plt.imshow(image_am)
plt.savefig('./results/feature_maps/BTR70/BTR70_am.PNG', bbox_inches='tight', dpi=300)
plt.show()
plt.imshow(image_ph)
plt.savefig('./results/feature_maps/BTR70/BTR70_ph.PNG', bbox_inches='tight', dpi=300)
plt.show()
plt.imshow(image_r)
plt.savefig('./results/feature_maps/BTR70/BTR70_r.PNG', bbox_inches='tight', dpi=300)
plt.show()
plt.imshow(image_i)
plt.savefig('./results/feature_maps/BTR70/BTR70_i.PNG', bbox_inches='tight', dpi=300)
plt.show()

image_r, image_i = torch.from_numpy(image_r).reshape(1,1,64,64), torch.from_numpy(image_i).reshape(1,1,64,64)  # 转化为tesor
image_r, image_i = image_r.to(device), image_i.to(device)                # 载入CUDA（）
image_r, image_i = image_r.type(torch.cuda.FloatTensor), image_i.type(torch.cuda.FloatTensor)


'''
train_dataset = MyDataset(img_r=traindata_r, img_i=traindata_i, label=trainlabel, transform=transforms.ToTensor())
test_dataset  = MyDataset(img_r=testdata_r, img_i=testdata_i, label=testlabel, transform=transforms.ToTensor())
print('real train data size: {} \nimaginary train data size: {}' \
      .format(train_dataset.img_r.shape[0], train_dataset.img_i.shape[0]))
print('real test data size: {} \nimaginary test data size: {}' \
      .format(test_dataset.img_r.shape[0], test_dataset.img_i.shape[0]))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
'''
#----------------model loading-----------------
model = torch.load(PATH)
# criterion = nn.NLLLoss()
# criterion = nn.CrossEntropyLoss()
# feature_list = ['Conv_A1', 'Conv_B1', 'Conv_C1','MaxPool2d_A2','MaxPool2d_B2','MaxPool2d_C2','Max_F1']

#----------------testing-----------------
model.eval()
with torch.no_grad():

    Conv_A1_r, Conv_A1_i = model.Conv_A1(image_r, image_i)
    x_r,             x_i = model.BN_A1(Conv_A1_r, Conv_A1_i)
    x_r,             x_i = model.ReLU_A1(x_r, x_i)
    Conv_A2_r, Conv_A2_i = model.Conv_A2(x_r, x_i)
    x_r,             x_i = model.BN_A2(Conv_A2_r, Conv_A2_i)
    x_r,             x_i = model.ReLU_A2(x_r, x_i)
    x_r_A2,       x_i_A2 = model.MaxPool2d_A2(x_r, x_i)
    
    Conv_B1_r, Conv_B1_i = model.Conv_B1(image_r, image_i)
    x_r,             x_i = model.BN_B1(Conv_B1_r, Conv_B1_i)
    x_r,             x_i = model.ReLU_B1(x_r, x_i)
    Conv_B2_r, Conv_B2_i = model.Conv_B2(x_r, x_i)
    x_r,             x_i = model.BN_B2(Conv_B2_r, Conv_B2_i)
    x_r,             x_i = model.ReLU_B2(x_r, x_i)
    x_r_B2,       x_i_B2 = model.MaxPool2d_B2(x_r, x_i)
    
    Conv_C1_r, Conv_C1_i = model.Conv_C1(image_r, image_i)
    x_r,             x_i = model.BN_C1(Conv_C1_r, Conv_C1_i)
    x_r,             x_i = model.ReLU_C1(x_r, x_i)
    Conv_C2_r, Conv_C2_i = model.Conv_C2(x_r, x_i)
    x_r,             x_i = model.BN_C2(Conv_C2_r, Conv_C2_i)
    x_r,             x_i = model.ReLU_C2(x_r, x_i)
    x_r_C2,       x_i_C2 = model.MaxPool2d_C2(x_r, x_i)
    
    MF1_r,         MF1_i = model.Concat_F1(x_r_A2, x_r_B2, x_r_C2, x_i_A2, x_i_B2, x_i_C2)
    
    Conv_A3_r, Conv_A3_i = model.Conv_A3(MF1_r, MF1_i)
    x_r,             x_i = model.BN_A3(Conv_A3_r, Conv_A3_i)
    x_r,             x_i = model.ReLU_A3(x_r, x_i)
    Conv_A4_r, Conv_A4_i = model.Conv_A4(x_r, x_i)
    x_r,             x_i = model.BN_A4(Conv_A4_r, Conv_A4_i)
    x_r,             x_i = model.ReLU_A4(x_r, x_i)
    x_r_A4,       x_i_A4 = model.MaxPool2d_A4(x_r, x_i)
    
    Conv_B3_r, Conv_B3_i = model.Conv_B3(MF1_r, MF1_i)
    x_r,             x_i = model.BN_B3(Conv_B3_r, Conv_B3_i)
    x_r,             x_i = model.ReLU_B3(x_r, x_i)
    Conv_B4_r, Conv_B4_i = model.Conv_B4(x_r, x_i)
    x_r,             x_i = model.BN_B4(Conv_B4_r, Conv_B4_i)
    x_r,             x_i = model.ReLU_B4(x_r, x_i)
    x_r_B4,       x_i_B4 = model.MaxPool2d_B4(x_r, x_i)
    
    Conv_C3_r, Conv_C3_i = model.Conv_C3(MF1_r, MF1_i)
    x_r,             x_i = model.BN_C3(Conv_C3_r, Conv_C3_i)
    x_r,             x_i = model.ReLU_C3(x_r, x_i)
    Conv_C4_r, Conv_C4_i = model.Conv_C4(x_r, x_i)
    x_r,             x_i = model.BN_C4(Conv_C4_r, Conv_C4_i)
    x_r,             x_i = model.ReLU_C4(x_r, x_i)
    x_r_C4,       x_i_C4 = model.MaxPool2d_C4(x_r, x_i)
    
    MF2_r,         MF2_i = model.Concat_F2(x_r_A4, x_r_B4, x_r_C4, x_i_A4, x_i_B4, x_i_C4)
    
    Conv_A5_r, Conv_A5_i = model.Conv_A5(MF2_r, MF2_i)
    x_r,             x_i = model.BN_A5(Conv_A5_r, Conv_A5_i)
    x_r,             x_i = model.ReLU_A5(x_r, x_i)
    x_r_A5,       x_i_A5 = model.MaxPool2d_A5(x_r, x_i)
    Conv_A6_r, Conv_A6_i = model.Conv_A6(x_r, x_i)
    x_r,             x_i = model.BN_A6(Conv_A6_r, Conv_A6_i)
    x_r,             x_i = model.ReLU_A6(x_r, x_i)
    x_r_A6,       x_i_A6 = model.AvgPool2d_A6(x_r, x_i)
    
    Conv_B5_r, Conv_B5_i = model.Conv_B5(MF2_r, MF2_i)
    x_r,             x_i = model.BN_B5(Conv_B5_r, Conv_B5_i)
    x_r,             x_i = model.ReLU_B5(x_r, x_i)
    x_r_B5,       x_i_B5 = model.MaxPool2d_B5(x_r, x_i)
    Conv_B6_r, Conv_B6_i = model.Conv_B6(x_r, x_i)
    x_r,             x_i = model.BN_B6(Conv_B6_r, Conv_B6_i)
    x_r,             x_i = model.ReLU_B6(x_r, x_i)
    x_r_B6,       x_i_B6 = model.AvgPool2d_B6(x_r, x_i)
    
    Conv_C5_r, Conv_C5_i = model.Conv_C5(MF2_r, MF2_i)
    x_r,             x_i = model.BN_C5(Conv_C5_r, Conv_C5_i)
    x_r,             x_i = model.ReLU_C5(x_r, x_i)
    x_r_C5,       x_i_C5 = model.MaxPool2d_C5(x_r, x_i)
    Conv_C6_r, Conv_C6_i = model.Conv_C6(x_r, x_i)
    x_r,             x_i = model.BN_C6(Conv_C6_r, Conv_C6_i)
    x_r,             x_i = model.ReLU_C6(x_r, x_i)
    x_r_C6,       x_i_C6 = model.AvgPool2d_C6(x_r, x_i)
    
    MF3_r,         MF3_i = model.Concat_F3(x_r_A6, x_r_B6, x_r_C6, x_i_A6, x_i_B6, x_i_C6)
    
    
    

    Conv_A1_r, Conv_A1_i = Conv_A1_r.data.cpu().numpy(), Conv_A1_i.data.cpu().numpy()
    Conv_A2_r, Conv_A2_i = Conv_A2_r.data.cpu().numpy(), Conv_A2_i.data.cpu().numpy()
    x_r_A2,       x_i_A2 = x_r_A2.data.cpu().numpy(), x_i_A2.data.cpu().numpy()

    Conv_B1_r, Conv_B1_i = Conv_B1_r.data.cpu().numpy(), Conv_B1_i.data.cpu().numpy()
    Conv_B2_r, Conv_B2_i = Conv_B2_r.data.cpu().numpy(), Conv_B2_i.data.cpu().numpy()
    x_r_B2,       x_i_B2 = x_r_B2.data.cpu().numpy(), x_i_B2.data.cpu().numpy()

    Conv_C1_r, Conv_C1_i = Conv_C1_r.data.cpu().numpy(), Conv_C1_i.data.cpu().numpy()
    Conv_C2_r, Conv_C2_i = Conv_C2_r.data.cpu().numpy(), Conv_C2_i.data.cpu().numpy()
    x_r_C2,       x_i_C2 = x_r_C2.data.cpu().numpy(), x_i_C2.data.cpu().numpy()
    
    MF1_r,         MF1_i = MF1_r.data.cpu().numpy(), MF1_i.data.cpu().numpy()

    scipy.io.savemat('./results/BTR70_ms.mat',{'A2_r' : x_r_A2, 'A2_i' : x_i_A2, 'B2_r':x_r_B2, 'B2_i':x_i_B2, 'C2_r':x_r_C2, 'C2_i':x_i_C2})
    print('feature maps saving start...')
    path = './results/feature_maps/BTR70/'
    fea_num = len(Conv_A1_r.squeeze())
    for i in range(fea_num):
        #------------------- stream-A---------------------
        ##   Conv_A1
        img_r = Conv_A1_r[0,i,:,:].squeeze()
        img_i = Conv_A1_i[0,i,:,:].squeeze()

        plt.imshow(img_r)
        savepath = path + 'Conv_A1_r/' + str(i) + '.png'
        plt.savefig(savepath, bbox_inches='tight', dpi=100)

        plt.imshow(img_i)
        savepath = path + 'Conv_A1_i/' + str(i) + '.png'
        plt.savefig(savepath, bbox_inches='tight', dpi=100)
        
        ##   Conv_A2
        img_r = Conv_A2_r[0,i,:,:].squeeze()
        img_i = Conv_A2_i[0,i,:,:].squeeze()
        
        plt.imshow(img_r)
        savepath = path + 'Conv_A2_r/' + str(i) + '.png'
        plt.savefig(savepath, bbox_inches='tight', dpi=100)

        plt.imshow(img_i)
        savepath = path + 'Conv_A2_i/' + str(i) + '.png'
        plt.savefig(savepath, bbox_inches='tight', dpi=100)
        
        ##   x_A2
        img_r = x_r_A2[0,i,:,:].squeeze()
        img_i = x_i_A2[0,i,:,:].squeeze()
        
        plt.imshow(img_r)
        savepath = path + 'xr_A2/' + str(i) + '.png'
        plt.savefig(savepath, bbox_inches='tight', dpi=100)

        plt.imshow(img_i)
        savepath = path + 'xi_A2/' + str(i) + '.png'
        plt.savefig(savepath, bbox_inches='tight', dpi=100)
        #------------------- stream-B---------------------
        ##   Conv_B1
        img_r = Conv_B1_r[0,i,:,:].squeeze()
        img_i = Conv_B1_i[0,i,:,:].squeeze()

        plt.imshow(img_r)
        savepath = path + 'Conv_B1_r/' + str(i) + '.png'
        plt.savefig(savepath, bbox_inches='tight', dpi=100)

        plt.imshow(img_i)
        savepath = path + 'Conv_B1_i/' + str(i) + '.png'
        plt.savefig(savepath, bbox_inches='tight', dpi=100)
        
        ##   Conv_B2
        img_r = Conv_B2_r[0,i,:,:].squeeze()
        img_i = Conv_B2_i[0,i,:,:].squeeze()
        
        plt.imshow(img_r)
        savepath = path + 'Conv_B2_r/' + str(i) + '.png'
        plt.savefig(savepath, bbox_inches='tight', dpi=100)

        plt.imshow(img_i)
        savepath = path + 'Conv_B2_i/' + str(i) + '.png'
        plt.savefig(savepath, bbox_inches='tight', dpi=100)
        
        ##   x_B2
        img_r = x_r_B2[0,i,:,:].squeeze()
        img_i = x_i_B2[0,i,:,:].squeeze()
        
        plt.imshow(img_r)
        savepath = path + 'xr_B2/' + str(i) + '.png'
        plt.savefig(savepath, bbox_inches='tight', dpi=100)

        plt.imshow(img_i)
        savepath = path + 'xi_B2/' + str(i) + '.png'
        plt.savefig(savepath, bbox_inches='tight', dpi=100)
        
        #------------------- stream-C---------------------
        ##   Conv_C1
        img_r = Conv_C1_r[0,i,:,:].squeeze()
        img_i = Conv_C1_i[0,i,:,:].squeeze()

        plt.imshow(img_r)
        savepath = path + 'Conv_C1_r/' + str(i) + '.png'
        plt.savefig(savepath, bbox_inches='tight', dpi=100)

        plt.imshow(img_i)
        savepath = path + 'Conv_C1_i/' + str(i) + '.png'
        plt.savefig(savepath, bbox_inches='tight', dpi=100)
        
        ##   Conv_C2
        img_r = Conv_C2_r[0,i,:,:].squeeze()
        img_i = Conv_C2_i[0,i,:,:].squeeze()
        
        plt.imshow(img_r)
        savepath = path + 'Conv_C2_r/' + str(i) + '.png'
        plt.savefig(savepath, bbox_inches='tight', dpi=100)

        plt.imshow(img_i)
        savepath = path + 'Conv_C2_i/' + str(i) + '.png'
        plt.savefig(savepath, bbox_inches='tight', dpi=100)
        
        ##   x_C2
        img_r = x_r_C2[0,i,:,:].squeeze()
        img_i = x_i_C2[0,i,:,:].squeeze()
        
        plt.imshow(img_r)
        savepath = path + 'xr_C2/' + str(i) + '.png'
        plt.savefig(savepath, bbox_inches='tight', dpi=100)

        plt.imshow(img_i)
        savepath = path + 'xi_C2/' + str(i) + '.png'
        plt.savefig(savepath, bbox_inches='tight', dpi=100)
        
        ##------------------ feature max valued fusion----------------
        img_r = MF1_r[0,i*3,:,:].squeeze()
        img_i = MF1_i[0,i*3,:,:].squeeze()
        
        plt.imshow(img_r)
        savepath = path + 'CCF1_r/' + str(i) + '.png'
        plt.savefig(savepath, bbox_inches='tight', dpi=100)

        plt.imshow(img_i)
        savepath = path + 'CCF1_i/' + str(i) + '.png'
        plt.savefig(savepath, bbox_inches='tight', dpi=100)                
        
        if (i+1) % 2 == 0:
            print('------Epoch [{} / {}], processing progress: {} %-------'.format(i+1, fea_num, (i+1)/fea_num * 100))
    print('feature maps save finished !!!')
    

    # fea_visual_tool = feature_visualization(model, feature_list)
    # feature_r, feature_i = fea_visual_tool(image_r, image_i)






























