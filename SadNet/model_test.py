import torch
from model.mul_intra import *
from model.att_net import *
from training import predicting, fusion_predicting ,collate
from GraphPairDataset import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from parse_data import Parser

def mul_batchsize():
    model_name = ['pocket_only_net','ami_only_net','pocket_and_ami','ami_att_net2_1.332_1.56_1.76_refined', 'pocket_att_net2_1.27_1.46_1.54_refined', 'ami_att_net', 'pocket_structure_net_1.332_chan',
                  'pocket_structure_net2_1.289', 'pocket_eazy_1.460']
    PATH = './model/saved_model/model_' + model_name[3] + '.model'
    #PATH = './model/saved_model/2020_model_pocket_only_net.model'
    device = 'cpu'#torch.device(0)
    model = ami_att_net2().to(device)
    model.load_state_dict(torch.load(PATH))
    test_data = GraphPairDataset(dataset='test', loc='./processed/多batch版本/refined_fast/')
    #test_loader = DataLoader(test_data, batch_size=1, shuffle=False,collate_fn=collate)
    test_loader2 = DataLoader(test_data, batch_size=16, shuffle=True,collate_fn=collate)
    #G1, P1 = predicting(model, device, test_loader)
    G, P = predicting(model, device, test_loader2)
    return G, P

def fusion_test():
    PATH = './model/saved_model/model_att_fusion2_1.252.model'
    checkpoint = torch.load(PATH)
    device = 'cpu'
    model1 = pocket_att_net2().to(device)
    model2 = ami_att_net2().to(device)
    model = att_fusion2().to(device)
    model1.load_state_dict(checkpoint['model1'])
    model2.load_state_dict(checkpoint['model2'])
    model.load_state_dict(checkpoint['model'])
    test_data = GraphPairDataset(dataset='test', loc='./processed/多batch版本/refined_fast/')
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False,collate_fn=collate)
    G, P = fusion_predicting(model1, model2, model, device, test_loader)
    #G, P = predicting(model1, device, test_loader)
    return G, P

def test_2020_pdb():
    code_name = list()
    li_name = list()
    prot_name = list()
    affinity = list()

    with open('./processed/data_list/input_list_2020.txt') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            code_name.append(line[0])
            li_name.append(line[3])
            prot_name.append(line[1])
            affinity.append(line[5])
    # print(code_name)
    process_data = None
    if os.path.exists('./processed/多batch版本/test_2020.pt'):
        pass
    else:
        process_data = Parser(cutoff=5, code_name=code_name, li_name=li_name, prot_name=prot_name, affinity=affinity,external_test_data=True)

    model_name = ['pocket_only_net', 'ami_only_net', 'pocket_and_ami_1.330']
    PATH = './model/saved_model/model_' + model_name[2] + '.model'
    device = 'cpu'#torch.device(0)
    model = pocket_and_ami().to(device)
    model.load_state_dict(torch.load(PATH))
    test_data = GraphPairDataset(data=process_data,dataset='test_2020')
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=collate)
    G, P = predicting(model, device, test_loader)
    return G, P

def fusion_2020_test():
    process_data = None
    if os.path.exists('./processed/多batch版本/test_2020.pt'):
        pass
    else:
        code_name = list()
        li_name = list()
        prot_name = list()
        affinity = list()

        with open('./processed/data_list/input_list_2020_refined.txt') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                code_name.append(line[0])
                li_name.append(line[3])
                prot_name.append(line[1])
                affinity.append(line[5])
        process_data = Parser(cutoff=5, code_name=code_name, li_name=li_name, prot_name=prot_name, affinity=affinity, external_test_data=True)

    PATH = './model/saved_model/model_fusion_cnn_1.307.model'
    checkpoint = torch.load(PATH)
    device = 'cpu'
    model1 = pocket_only_net().to(device)
    model2 = ami_only_net2().to(device)
    model = fusion().to(device)
    model1.load_state_dict(checkpoint['model1'])
    model2.load_state_dict(checkpoint['model2'])
    model.load_state_dict(checkpoint['model'])
    test_data = GraphPairDataset(data=process_data,dataset='test_2020')
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=collate)
    G, P = fusion_predicting(model1, model2, model, device, test_loader)
    #G, P = predicting(model2, device, test_loader)
    return G, P

#G, P = mul_batchsize()
G, P = fusion_test()
#G, P = test_2020_pdb()
#G, P = fusion_2020_test()
print('rmse={:.6},mse={:.6},pearson={:.6},mae={:.6}'.format(rmse(G,P),mse(G,P),pearson(G,P),mae(G,P)))
delta = G - P
a, b, c, d = 0, 0, 0, 0
for i in delta:
    if abs(i)<=0.5:
        a += 1
    elif abs(i)>0.5 and abs(i)<=1.5:
        b += 1
    elif abs(i)>1.5 and abs(i)<=2.5:
        c += 1
    else:
        d += 1
print('less than 0.5: {}\nbetween 0.5 and 1.5: {}\nbetween 1.5 and 2.5: {}\nlager than 2.5: {}'.format(a, b, c, d))

#myfig = plt.g1cf()
plt.scatter(G,P)
plt.title('rmse={:.4},mse={:.4},pearson={:.4},mae={:.4}'.format(rmse(G,P),mse(G,P),pearson(G,P),mae(G,P)), fontsize=12)
parameter = np.polyfit(G, P, 1)
y2 = parameter[0] * G + parameter[1]
plt.plot(G, y2, color='g') #一阶拟合线

y3 = G
plt.plot(G, y3, color='r')
plt.xlabel("True", fontsize=16)
plt.ylabel("Predict", fontsize=16)
plt.savefig('./feature/core', dpi=600)
#w = 150/ 10/ 2.54
#h = 150/ 10/ 2.54
#plt.figure(figsize=(w, h))
#myfig.savefig('output0(without h2)2020.tif', dpi=600)#, pil_kwargs={'compression': 'tiff_lzw'})

plt.show()


