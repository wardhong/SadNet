from model.mul_intra import *
from training import predicting, fusion_predicting ,collate
from GraphPairDataset import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from parse_data import Parser

def fusion_test():
    PATH = './model/saved_model/model_fusion2_1.281.model'
    checkpoint = torch.load(PATH)
    device = 'cpu'
    model1 = pocket_only_net().to(device)
    model2 = ami_only_net().to(device)
    model = fusion2().to(device)
    model1.load_state_dict(checkpoint['model1'])
    model2.load_state_dict(checkpoint['model2'])
    model.load_state_dict(checkpoint['model'])
    test_data = GraphPairDataset(dataset='test')
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False,collate_fn=collate)

    model1.eval()
    model2.eval()
    model.eval()
    i = 0
    while i < 512:
        total_preds = torch.Tensor()
        total_labels = torch.Tensor()
        print('打乱第{}个特征'.format(i))
        with torch.no_grad():
            for data in test_loader:
                data0 = data[0].to(device)
                data1 = data[1].to(device)
                data2 = data[2].to(device)
                data3 = data[3].to(device)
                output1, h1 = model1(data0, data1, data2, data3)
                output2, h2 = model2(data0, data1, data2, data3)

                #idx = torch.randperm(h1.size(0))

                a = []
                for j in range(h1.size(0)):
                    a.append(0)

                #H = h1.clone()

                #h1[:, i] = T[idx, i]
                for m in range(16):                   #一位一位把16位置零
                    if i + m < 512:
                        h2[:, i + m] = torch.tensor(a)

                #h2 = torch.zeros(h2.shape)
                output = model(h1, h2)
                total_preds = torch.cat((total_preds, output.cpu()), 0)  # 把每轮的output数据叠加在一起
                total_labels = torch.cat((total_labels, data0.y.view(-1, 1).cpu()), 0)
        G, P = total_labels.numpy().flatten(), total_preds.numpy().flatten()
        print('rmse={},ci={},pearson={},SD={},mae={}'.format(rmse(G, P), ci(G, P), pearson(G, P), sd(G, P), mae(G, P)))
        out = open('result_feature_importance_h2.xls','a',encoding='gbk')
        out.write(str(rmse(G, P)))
        out.write('\t')
        out.write(str(ci(G, P)))
        out.write('\t')
        out.write(str(pearson(G, P)))
        out.write('\t')
        out.write(str(sd(G, P)))
        out.write('\t')
        out.write(str(mae(G, P)))
        out.write('\n')
        i = i + 16
        #return total_labels.numpy().flatten(), total_preds.numpy().flatten()

fusion_test()
