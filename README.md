# IEEE TGRS---MS-CVNets-a-novel-SAR-ATR-method-based-on-Multi-Stream-Complex-Valued-Networks

The code in this toolbox implements the "SAR Automatic Target Recognition Method based on Muliti-Stream Complex-Valued Networks" in IEEE Transactions on Geoscience and Remote Sensing (TGRS). More specifically, it is detailed as follow.

![MS-CVNets框架5](https://user-images.githubusercontent.com/44805578/169686156-a2f9dfa9-ce57-4659-8884-bc3ec90c6e5b.png)

# usage
## Complex-MSTAR dataset
this Complex-MSTAR dataset is based on the original MSTAR program. we do not participate in the data acquisition work, only data redistribution and collation. the complex-mstar dataset structure is as follows:

---Complex-MSTAR-<br>
-----------------data_SOC-class10-trian-imgs<br>
----------------------------------------data_train_64.mat<br>
----------------------------------------data_train_128.mat<br>
----------------------------------test-data_test_64.mat<br>
---------------------------------------data_test_128.mat<br>
-------------------------calss3-trian--data_train_64.mat<br>
---------------------------------------data_train_128.mat<br>
----------------------------------test-data_test_64.mat<br>
---------------------------------------data_test_128.mat<br>
-----------------data_EOC-depression_variation<br>
------------------------ -noise varision<br>
--------------------------version variation<br>
the Complex-MSTAR dataset provide both size of 64x64 and 128x128 for different task requriements.

# requriements
---python 3.7  
---pytorch 1.6  
---CUDA 10.1  

# Training

-             python Train.py                        

# Testing

-              python Test.py                 


# Citation
please kindly cite this paper if our MS-CVNets can give you any inspiration for your research, thanks a lot.  

Z. Zeng, J. Sun, Z. Han and W. Hong, "SAR Automatic Target Recognition Method Based on Multi-Stream Complex-Valued Networks," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-18, 2022, Art no. 5228618, doi: 10.1109/TGRS.2022.3177323.

# Contact
Zhiqiang Zeng  
Email:zengzq@buaa.edu.cn  

# references
1. the original MSTAR dataset information:  https://www.sdms.afrl.af.mil/
2. we would like to appreciate the ChihebTrabelsi, wavefrontshaping and ivannz in the help of coding basis. we bulid the MS-CVNets principal framework on their basis.<br>
3. https://github.com/ivannz/cplxmodule  
4. https://github.com/wavefrontshaping/complexPyTorch  
5. https://github.com/ChihebTrabelsi/deep_complex_networks                            


