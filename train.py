import argparse
import os
import sys
import yaml
from data.dataset import BirdCallDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from utils.logging import Averager
from modules.ResNet import resnet152, resnet50
from utils.loss import AsymmetricLoss
from utils.ema import ModelEma
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import time
import warnings
import torch.nn.functional as F
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore") 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Bird Call Classifier Training Session"
    )
    
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        help="Define training epochs",
        default=500
    )
    
    parser.add_argument(
        "-s",
        "--saved",
        type=bool,
        help="Determine whether Load model from saved or not",
        default=False
    )
    
    parser.add_argument(
        "-f",
        "--file_name",
        type=str,
        help="File name for saving trained model",
        default=f"{time.time()}"
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    with open('./config/train.yml') as f:
        conf = yaml.load(f,Loader=yaml.FullLoader)
    
    epochs = args.epochs
 
    if not os.path.exists(conf['train']['saved_model']) and args.saved:
        raise FileNotFoundError('No such saved model {}'.format(conf['train']['saved_model']))
    
    if not os.path.exists(conf['train']['saved_model']):
        os.makedirs(conf['train']['saved_model'])
    
    #model = Model(conf['model'])
    
    # init model
    #model = resnet152(num_classes=conf['train']['num_classes'])
    model = resnet50(num_classes=conf['train']['num_classes'])
    model.to(device)
    ema = ModelEma(model, 0.9997)

    # get parameter index with grad
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    
    
    # training parameter assign
    learning_rate=conf['train']['learning_rate']
    batch_size = conf['train']['batch_size']
    epochs = conf['train']['epochs']
    
    # prepare datasets
    dataset = BirdCallDataset(conf['data_folder'])
    dataloader = DataLoader(dataset,batch_size=conf['train']['batch_size'],shuffle=True)
    
    # init utility classes
    loss_avg = Averager()
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    scaler = GradScaler()
    optimizer = optim.Adam(filtered_parameters, lr=learning_rate, betas=(0.9, 0.999))
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(dataset), epochs=epochs,
                                        pct_start=0.2)
    
    best_loss = 100
    cur_time = time.time()
    # Run Training Session
    print(len(dataset))
    with tqdm(range(epochs),unit="epoch") as tepoch:
        for epoch in tepoch:
            model.train()
            for batch,data in enumerate(dataloader):
                tepoch.set_description(f" Epoch {epoch+1}/{batch} ")
                
                wav, bird = data
                
                wav = wav.to(torch.float32)
                wav = wav.to(device)
                bird_smooth = np.where(bird==1,0.995,0.0025)
                bird_smooth = torch.from_numpy(bird_smooth).to(device)
                
                with autocast():  # mixed precision
                    output = model(wav).float()  # sigmoid will be done in loss !

                loss = criterion(output, bird_smooth)
                loss_avg.add(loss)
                model.zero_grad()

                scaler.scale(loss).backward()
                # loss.backward()

                scaler.step(optimizer)
                scaler.update()
                # optimizer.step()

                scheduler.step()
                ema.update(model)
                
                pred_score = torch.where(F.softmax(output,dim=1)>conf['train']['threshold'],1,0)
                t1 = pred_score.cpu().detach().numpy()[0]
                t2 = bird.cpu().detach().numpy()[0]
                
                torch.nn.utils.clip_grad_norm_(model.parameters(),5)  # gradient clipping with 5 (Default)
                #https://kh-kim.gitbook.io/natural-language-processing-with-pytorch/00-cover-6/05-gradient-clipping
                tepoch.set_postfix(loss=loss_avg.val().item(), f1_score=f1_score(t1,t2))
                
                del wav, bird, loss, output, t1,t2,pred_score
                
            if loss_avg.val().item() < best_loss:
                best_loss = loss_avg.val().item()
                torch.save(model.state_dict(),os.path.join(conf['train']['save_folder'],f'{args.file_name}.pth'))
            # validation section, ToDo.
            '''
            model.eval()
            with torch.no_grad():
                pass
            '''
            
if __name__ == "__main__":
    main()
