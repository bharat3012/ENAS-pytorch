
import torch
from src import utils
import torch.nn.functional as F
import pandas as pd
import numpy as np
#import scipy.misc
#import matplotlib
#matplotlib.use('Agg')
#from PIL import Image
#import cv2
#import matplotlib.pyplot as plt
use_cuda = torch.cuda.is_available()
input = pd.read_csv('test_new.csv')
def predict(i):
    model = torch.load('plots/Comphuman_comp_char_cnn_epoch_None_6_0.001_loss_0.379_acc_0.8528.pth', map_location = 'cpu')#2000
        #cha_cnn_epoch_None_8_0.001_loss_0.6073_acc_0.6645.pth
    #bn_char_cnn_epoch_None_2_0.001_loss_0.6751_acc_0.6468.pth ---- best
    #bn_char_cnn_epoch_None_6_0.00025_loss_0.5957_acc_0.7647.pth
    #Better_char_cnn_epoch_None_2_0.001_loss_0.6283_acc_0.6556.pth
    #Better_char_cnn_epoch_None_6_0.00025_loss_0.5098_acc_0.7555.pth   ----Best 13 acc 81
    #char_cnn_epoch_None_5_0.001_loss_0.5721_acc_0.7348.pth   ----- 16 but acc 76
    #char_cnn_epoch_None_5_0.001_loss_0.5721_acc_0.7348.pth
    #5000- human_char_cnn_epoch_None_9_0.001_loss_0.4701_acc_0.8337.pth
    #2000 - human_char_cnn_epoch_None_9_0.001_loss_0.4183_acc_0.8492.pth
    #mouse_2000_char_cnn_epoch_None_8_0.001_loss_0.5974_acc_0.7657.pth
    # complete -- mouse_2000_char_cnn_epoch_None_9_0.001_loss_0.967_acc_0.5129.pth
#    activation = {}
#    def get_activation(name):
#        def hook(model, input, output):
#            activation[name] = output.detach()
#        return hook
    

    processed_input = utils.preprocess_input(input['sequences'][i])
    processed_input = torch.tensor(processed_input)
    processed_input = processed_input.unsqueeze(0)
    if use_cuda:
        processed_input = processed_input.to('cuda')
        model = model.to('cuda')
    #model.conv_layers[0].register_forward_hook(get_activation('conv_layers[0]'))
    #model.eval()     
    #model = models.model(pretrained=True, aux_logits=False) 
    
    prediction = model(processed_input)
    #print("predict", prediction)
    probabilities = F.softmax(prediction, dim=1)
    _, preds = torch.max(probabilities, 1)
    
    
#    act = activation['conv_layers[0]'].squeeze()
#    #print(act.shape)
#    act = act.cpu()
#    #print(act.shape)
#    act = np.expand_dims(act, axis=0)
#    act = np.squeeze(act, axis=0)
#    #print(act.shape)
    
#    pred = preds.numpy()
    #b = (act - np.min(act))/np.ptp(act)

# Normalised [0,255] as integer
    #act = 255*(act - np.min(act))/np.ptp(act).astype(int)
#    plt.imshow(act) 
## Normalised [-1,1]
#   # d = 2*(act - np.min(act))/np.ptp(act)-1
#    if pred == 1:
#        plt.savefig('l1/1stlayer'+  str(i)+ '.png')
#    else:
#        plt.savefig('l10/1stlayer'+  str(i)+ '.png')
    
#    if pred ==1:
#        scipy.misc.imsave('l1/1stlayer'+  str(i)+ '.png', act)
#    else:
#        scipy.misc.imsave('l10/1stlayer'+  str(i)+ '.png', act)
#    
    return preds, probabilities
test_acc =0.0
count = 0

for i in range(len(input)):
    
    if __name__ == "__main__":

         prediction, probabilities = predict(i)
#         layer1 = k[0].numpy()
#         layer2 = k[1].numpy()
#         layer3 = k[2].numpy()
#         layer4 = k[3].numpy()
#         layer5 = k[4].numpy()
#         layer6 = k[5].numpy()
#         layer7 = k[6].numpy()
#         layer8 = k[7].numpy()
#         layer9 = k[8].numpy()
         
         
         #test_acc += torch.sum(prediction==input['target'][i])
         #print("test_acc=", test_acc)         
         #print('input : {}'.format(args.text))
         #print('prediction : {}, target : {}, test_acc : {}'.format(prediction, input['target'][i], test_acc))
         print('prediction : {}, prob : {} '.format(prediction, probabilities))
#         predm = prediction.numpy()
#         if predm ==1:
#             print('num : {}, prediction : {}, probabilty: {}'.format(i, prediction, probabilities)) 
#         if (prediction.data.cpu().numpy() == 0):
#             count = count+1
#         print("Count 1 misclassified :", count)     

         #data, _ = dataset[0]
         #data.unsqueeze_(0)
         #output = model(data)
        
#         act = activation['conv1'].squeeze()
#         fig, axarr = plt.subplots(act.size(0))
#         for idx in range(act.size(0)):
#             axarr[idx].imshow(act[idx])
#print("Test Accuracy : ", 100*(test_acc.data.cpu().numpy()/len(input) ))