import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
#from model.model import *
import torch.backends.cudnn as cudnn
import random
from model.mul_intra import *
from parse_data import Parser
from GraphPairDataset import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model.att_net import *
# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    num = 0
    for batch_idx, data in enumerate(train_loader):  # train_loader里包含了所有数据集，每次迭代出来一组，一组512个数据，20个组输出一次
        #data0 = data[0].to(device)
        data1 = data[0].to(device)
        data2 = data[1].to(device)
        data3 = data[2].to(device)
        optimizer.zero_grad()
        output, h = model(data1, data2, data3)
        loss = loss_fn(output, data1.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        num += len(data1.y)
        # print(len(data.y))
        if (batch_idx + 1) % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           num,
                                                                           len(train_loader.dataset),
                                                                           100. * (batch_idx + 1) / len(train_loader),
                                                                           loss.item()))

# training function at each epoch
def fusion_train(model1, model2, model, device, train_loader, optimizer, optimizer1,optimizer2, epoch, mode):
    if mode == 'nopre_train':
        print('Training on {} samples...'.format(len(train_loader.dataset)))
        model1.train()
        model2.train()
        model.train()
        num = 0
        for batch_idx, data in enumerate(train_loader):  # train_loader里包含了所有数据集，每次迭代出来一组，一组512个数据，20个组输出一次
            data0 = data[0].to(device)
            data1 = data[1].to(device)
            data2 = data[2].to(device)
            data3 = data[3].to(device)
            optimizer.zero_grad()
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            output1, h1 = model1(data0, data1, data2, data3)
            output2, h2 = model2(data0, data1, data2, data3)
            h1 = h1.detach()
            h2 = h2.detach()
            output = model(h1, h2)
            loss1 = loss_fn(output1, data0.y.view(-1, 1).float().to(device))
            loss2 = loss_fn(output2, data0.y.view(-1, 1).float().to(device))
            loss = loss_fn(output, data0.y.view(-1, 1).float().to(device))
            loss1.backward(retain_graph=True)
            loss2.backward(retain_graph=True)
            loss.backward()
            optimizer.step()
            optimizer1.step()
            optimizer2.step()
            num += len(data0.y)
            # print(len(data.y))
            if (batch_idx + 1) % LOG_INTERVAL == 0:
                print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\tLoss2: {:.6f}\tLoss: {:.6f}'.format(epoch,
                                                                               num,
                                                                               len(train_loader.dataset),
                                                                               100. * (batch_idx + 1) / len(train_loader),
                                                                               loss1.item(),loss2.item(),loss.item()))
    else:
        print('Training on {} samples...'.format(len(train_loader.dataset)))
        model.train()
        num = 0
        for batch_idx, data in enumerate(train_loader):  # train_loader里包含了所有数据集，每次迭代出来一组，一组512个数据，20个组输出一次
            data1 = data[0].to(device)
            data2 = data[1].to(device)
            data3 = data[2].to(device)
            optimizer.zero_grad()
            output1, h1 = model1(data1, data2, data3)
            output2, h2 = model2(data1, data2, data3)
            output = model(h1, h2)
            loss = loss_fn(output, data1.y.view(-1, 1).float().to(device))
            loss.backward()
            optimizer.step()
            num += len(data1.y)
            # print(len(data.y))
            if (batch_idx + 1) % LOG_INTERVAL == 0:
                print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,num,len(train_loader.dataset),100. * (batch_idx + 1) / len(train_loader),loss.item()))


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            #data0 = data[0].to(device)
            data1 = data[0].to(device)
            data2 = data[1].to(device)
            data3 = data[2].to(device)
            output, h = model(data1,data2,data3)
            #print(output)
            # print(output.shape)
            total_preds = torch.cat((total_preds, output.cpu()), 0)  # 把每轮的output数据叠加在一起
            total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)
            # print(total_preds.shape)
            # print(total_preds.numpy().flatten())
            # print(total_labels)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

def fusion_predicting(model1, model2, model, device, loader):
    model1.eval()
    model2.eval()
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data1 = data[0].to(device)
            data2 = data[1].to(device)
            data3 = data[2].to(device)
            output1, h1 = model1(data1, data2, data3)
            output2, h2 = model2(data1, data2, data3)
            #h2 = torch.zeros(h1.size())
            output = model(h1, h2)
            total_preds = torch.cat((total_preds, output.cpu()), 0)  # 把每轮的output数据叠加在一起
            total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)

            # print(total_preds.shape)
            # print(total_preds.numpy().flatten())
            # print(total_labels)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def adjust_learning_rate(optimizer, epoch, start_lr):
    lr = start_lr * (0.7 ** (epoch // 50)) #0.7
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('lr:{}'.format(lr))

def collate(data_list):
    #batch0 = Batch.from_data_list([data[0] for data in data_list])
    batch1 = Batch.from_data_list([data[0] for data in data_list])
    batch2 = Batch.from_data_list([data[1] for data in data_list])
    batch3 = Batch.from_data_list([data[2] for data in data_list])
    return batch1, batch2, batch3

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    seed_torch(41)
    code_name = list()
    li_name1 = list()
    li_name2 = list()
    prot_name1 = list()
    prot_name2 = list()
    affinity = list()

    with open('./processed/data_list/input_list_2020_1405.txt') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            code_name.append(line[0])
            li_name1.append(line[3])
            li_name2.append(line[4])
            prot_name1.append(line[1])
            prot_name2.append(line[2])
            affinity.append(line[5])
    # print(code_name)
    process_data = None
    loc = ['./processed/多batch版本/general_2013/', './processed/多batch版本/refined_fast/', './processed/多batch版本/']
    loc = loc[1]
    if os.path.exists(loc+'train.pt'):
        #process_data = Parser(cutoff=5,code_name=code_name,li_name=li_name1,li_name2=li_name2,prot_name=prot_name1,prot_name2=prot_name2,affinity=affinity)
        pass
    else:
        process_data = Parser(cutoff=5,code_name=code_name,li_name=li_name1,li_name2=li_name2,prot_name=prot_name1,prot_name2=prot_name2,affinity=affinity)#,external_test_data='2020_1405')


    cuda_name = 0
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    print(device)
    TRAIN_BATCH_SIZE = 64
    VALID_BATCH_SIZE = 64
    TEST_BATCH_SIZE = 64
    LR = 0.0001#0.00005

    LR1 = 0.001
    LR2 = 0.001
    LOG_INTERVAL = 10
    NUM_EPOCHS = 300
    loss_fn = nn.MSELoss(reduction='mean')
    Loss = []
    Loss_test = []
    Loss_2020 = []
    Loss_2013 = []
    best_mse = 100
    best_rmse = 100
    best_ci = 0
    best_pearson = 0
    best_epoch = -1

    # Main program: iterate over different datasets
    train_data = GraphPairDataset(data=process_data, dataset='train', loc=loc)
    vaild_data = GraphPairDataset(data=process_data, dataset='valid', loc=loc)
    test_data = GraphPairDataset(data=process_data, dataset='test', loc=loc)
    test_data2013 = GraphPairDataset(data=process_data, dataset='test_2013', loc=loc)
    test_data2020 = GraphPairDataset(data=process_data, dataset='test_2020_fast', loc=loc)

    # make data PyTorch mini-batch processing ready
    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate, drop_last=True)
    valid_loader = DataLoader(vaild_data, batch_size=VALID_BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_2013_loader = DataLoader(test_data2013, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_2020_loader = DataLoader(test_data2020, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate)
    print(torch.cuda.is_available())
    print('Learning rate: ', LR)
    print('Epochs: ', NUM_EPOCHS)

    training_mode = 'nofusion'
    mode = 'pre-train'
    if training_mode == 'fusion':
        modeling = att_fusion2#att_fusion2
        model_st = modeling.__name__
        model1 = pocket_att_net2().to(device)
        model2 = ami_att_net2().to(device)
        model = modeling().to(device)
        if mode == 'pre-train':
            PATH1 = './model/saved_model/model_pocket_att_net2_1.27_1.46_1.54_refined.model'#'./model/saved_model/model_pocket_only_net_1.39.model'
            PATH2 = './model/saved_model/model_ami_att_net2_1.332_1.56_1.76_refined.model'#'./model/saved_model/model_only_att_net_1.314.model'
            #path3 = './model/saved_model/model_att_fusion2_1.171_1.466_general.model'
            model1.load_state_dict(torch.load(PATH1))
            model2.load_state_dict(torch.load(PATH2))
            #model.load_state_dict(torch.load(path3)['model'])
        optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': LR}])

        optimizer1 = torch.optim.Adam([{'params': model1.parameters(), 'lr': LR1}])

        optimizer2 = torch.optim.Adam([{'params': model2.parameters(), 'lr': LR2}])

        model_file_name = 'model_' + model_st + '.model'
        result_file_name = 'result_' + model_st + '.csv'
        for epoch in range(NUM_EPOCHS):
            adjust_learning_rate(optimizer, epoch, LR)
            adjust_learning_rate(optimizer1,epoch, LR1)
            adjust_learning_rate(optimizer2,epoch, LR2)
            fusion_train(model1, model2, model, device, train_loader, optimizer, optimizer1, optimizer2, epoch + 1, mode)

            G, P = fusion_predicting(model1, model2, model, device, test_loader)
            RMSE = rmse(G, P)
            Loss_test.append(RMSE)
            print('testset:rmse={:.6},mse={:.6},pearson={:.6},mae={:.6}'.format(rmse(G, P), mse(G, P), pearson(G, P),
                                                                                mae(G, P)))

            G_2020, P_2020 = fusion_predicting(model1, model2, model, device, test_2020_loader)
            Loss_2020.append(rmse(G_2020, P_2020))
            print('test_2020:rmse={:.6},mse={:.6},pearson={:.6},mae={:.6}'.format(rmse(G_2020, P_2020), mse(G_2020, P_2020), pearson(G_2020, P_2020),
                                                                                mae(G_2020, P_2020)))
            G_2013, P_2013 = fusion_predicting(model1, model2, model, device, test_2013_loader)
            Loss_2013.append(rmse(G_2013, P_2013))
            print('test_2013:rmse={:.6},mse={:.6},pearson={:.6},mae={:.6}'.format(rmse(G_2013, P_2013),
                                                                                  mse(G_2013, P_2013),
                                                                                  pearson(G_2013, P_2013),
                                                                                  mae(G_2013, P_2013)))
            # 返回验证集集的标签和误差
            G_val, P_val = fusion_predicting(model1, model2, model, device, valid_loader)
            Loss.append(rmse(G_val, P_val))

            ret = [rmse(G_val, P_val), mse(G_val, P_val), pearson(G_val, P_val), spearman(G_val, P_val), ci(G_val, P_val)]
            if ret[1] < best_mse:
                state = {'model1': model1.state_dict(),
                         'model2': model2.state_dict(),
                         'model': model.state_dict()}
                torch.save(state, './model/saved_model/' + model_file_name + '_' + str(RMSE))
                with open(result_file_name, 'w') as f:
                    f.write(','.join(map(str, ret)))
                best_epoch = epoch + 1
                best_mse = ret[1]
                best_pearson = ret[2]
                best_rmse = ret[0]

                print('rmse improved at epoch ', best_epoch, '; best_rmse,best_mse,best_pearson:', best_rmse, best_mse,
                      best_pearson, model_st)
            else:
                print(ret[0], 'No improvement since epoch ', best_epoch, '; best_rmse,best_mse,best_pearson:',
                      best_rmse, best_mse, best_pearson, model_st)

    else: #单独训练口袋模块或者序列模块
        modeling = pocket_att_net2
        model_st = modeling.__name__
        model = modeling().to(device)
        #model.load_state_dict(torch.load('./model/saved_model/model_pocket_att_net2_1.183_general.model')) #1.21_1.52_1.48
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        model_file_name = 'model_' + model_st + '.model'
        #result_file_name = 'result_' + model_st + '.csv'
        # training the model
        for epoch in range(NUM_EPOCHS):
            adjust_learning_rate(optimizer, epoch, LR)

            train(model, device, train_loader, optimizer, epoch + 1)
            G_core, P_core = predicting(model, device, test_loader)
            RMSE = rmse(G_core, P_core)
            Loss_test.append(RMSE)
            print('testset:rmse={:.6},mse={:.6},pearson={:.6},mae={:.6}'.format(rmse(G_core, P_core), mse(G_core, P_core), pearson(G_core, P_core), mae(G_core, P_core)))

            G_2020, P_2020 = predicting(model, device, test_2020_loader)
            Loss_2020.append(rmse(G_2020, P_2020))
            print('test_2020:rmse={:.6},mse={:.6},pearson={:.6},mae={:.6}'.format(rmse(G_2020, P_2020),
                                                                                  mse(G_2020, P_2020),
                                                                                  pearson(G_2020, P_2020),
                                                                                  mae(G_2020, P_2020)))
            G_2013, P_2013 = predicting(model, device, test_2013_loader)
            Loss_2013.append(rmse(G_2013, P_2013))
            print('test_2013:rmse={:.6},mse={:.6},pearson={:.6},mae={:.6}'.format(rmse(G_2013, P_2013),
                                                                                  mse(G_2013, P_2013),
                                                                                  pearson(G_2013, P_2013),
                                                                                  mae(G_2013, P_2013)))

            G, P = predicting(model, device, valid_loader)  # 返回验证集集的标签和误差
            Loss.append(rmse(G, P))
            ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]
            if ret[1] < best_mse:
                torch.save(model.state_dict(), './model/saved_model/' + model_file_name + '_' + str(RMSE))
                #with open(result_file_name, 'w') as f:
                #    f.write(','.join(map(str, ret)))
                best_epoch = epoch + 1
                best_mse = ret[1]
                best_pearson = ret[2]
                best_rmse = ret[0]
                print('rmse improved at epoch ', best_epoch, '; best_rmse,best_mse,best_pearson:', best_rmse, best_mse,
                      best_pearson, model_st)
            else:
                print(ret[0], 'No improvement since epoch ', best_epoch, '; best_rmse,best_mse,best_pearson:', best_rmse,
                      best_mse, best_pearson, model_st)

    fig, ax = plt.subplots()
    x = list(range(NUM_EPOCHS+1))
    x.pop(0)
    ax.plot(x, Loss, 'b', label='val')
    ax.plot(x, Loss_2013, 'g', label='test_2013')
    ax.plot(x, Loss_test, 'r', label = 'test')
    ax.plot(x, Loss_2020, 'y', label = 'test_2020')
    ax.set_xlabel('epoch')
    ax.set_ylabel('rmse')
    ax.set_title('learning rate:{}'.format(LR))
    ax.legend()
    plt.show()
