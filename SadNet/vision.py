# coding: utf-8
import os
import ast
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from model.att_net import *
import torch
import torch.autograd as autograd
import torchvision.transforms as transforms
from training import *
import torchinfo
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

features_in_hook, features_out_hook = [], []
# 为了读取模型中间参数变量的梯度而定义的辅助函数
def extract(g):
    global features_grad
    features_grad = g

def hook(module, fea_in, fea_out):
    features_in_hook.append(fea_in)
    features_out_hook.append(fea_out)
    return None

def get_data(pdb_id, ligand, prot, affi):
    '''
    dir_list = ['./raw_data/casf2013/', './raw_data/PDBbind_v2020_refined/refined-set/', './raw_data/general-set-except-refined/', './raw_data/refined-set/']
    for i in dir_list:
        pdb_path = i + pdb_id[0]
        if os.path.exists(pdb_path):
    '''
    process_data = Parser(cutoff=5, code_name=pdb_id, li_name2=ligand,prot_name2=prot, affinity=affi, external_test_data=pdb_id[0])
    test_data = GraphPairDataset(root='./processed/多batch版本/single_test/', data=process_data, dataset=pdb_id[0], loc='./processed/多batch版本/single_test/')
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate)
    #break
    return test_loader

def draw_CAM(D, visual_heatmap=False, out_layer=None, model=None, model1=None, model2=None, mol=None):
    num_atom = mol.GetNumAtoms()
    data1 = D[0].to(device)
    data2 = D[1].to(device)
    data3 = D[2].to(device)
    label = data1.y
    model.eval()
    # model转为eval模式
    if model1!=None:
        model1.eval()
        model2.eval()
        output1, h1 = model1(data1, data2, data3)
        output2, h2 = model2(data1, data2, data3)
        for (name, module) in model.named_modules():
            if name in out_layer:
                module.register_forward_hook(hook=hook)
        output = model(h1, h2)
        hidd = features_in_hook[0][0]

    else:
        for (name, module) in model.named_modules():
            if name in out_layer:
                module.register_forward_hook(hook=hook)
        output, h = model(data1, data2, data3)
        hidd = features_in_hook[0][0]  #conv1层用输入前特征
    if out_layer == 'conv1':
        features_grad = autograd.grad(output, hidd, allow_unused=True)[0]
        grads = features_grad.squeeze()
        #grads = features_grad.repeat(20, 1)#[0]  # 获取梯度  # 此处batch size默认为1，所以去掉了第0维（batch size维）

        # 计算heatmap
        heatmap = grads.detach().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap = np.sum(heatmap, axis=1)
        heatmap /= np.max(heatmap)

        heatmap = heatmap[:num_atom]
        weight = heatmap
        heatmap = np.tile(weight, (5, 1))

        # 可视化原始热力图
        if visual_heatmap:
            heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
            plt.matshow(heatmap, cmap=plt.cm.Blues)
            plt.show()
        id = []
        r = {}
        for i in range(len(weight)):
            if weight[i] > 0.005:
                id.append(i)
                r[i] = float(str(weight[i].round(2)))
        d = Draw.rdMolDraw2D.MolDraw2DCairo(1200, 1200)
        Draw.rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=id, highlightAtomRadii=r)
        d.FinishDrawing()
        d.WriteDrawingText("./rdMolDraw2D_output_" + pdb_id + "2.png")
    elif out_layer == 'ica1.fc11':
        features_grad = autograd.grad(output, hidd, allow_unused=True)[0]
        features_grad = features_grad.view(1, 79, 8, 64).permute(0, 2, 1, 3)[0]
        heatmap = features_grad.detach().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap = np.sum(heatmap, axis=1)
        heatmap /= np.max(heatmap)
        #heatmap = np.sum(heatmap, axis=1)
        #heatmap = np.tile(heatmap, (5, 1))
        heatmap1 = np.sum(heatmap, axis=1).reshape((8, 1))
        heatmap1 /= np.max(heatmap1)
        heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
        heatmap1 = np.uint8(255 * heatmap1)
        # 可视化原始热力图
        if visual_heatmap:
            fig, ax = plt.subplots(figsize=(30, 15))
            plt.axis('off')
            ax.matshow(heatmap, cmap=plt.cm.Blues)
            plt.savefig('./feature/channel64', dpi=1000)

            fig2, ax2 = plt.subplots(figsize=(5, 10))
            ax2.matshow(heatmap1, cmap=plt.cm.Blues)
            plt.axis('off')
            plt.savefig('./feature/channel8', dpi=600)
            plt.show()
    elif out_layer in ['fc00', 'fc01']:
        features_grad = autograd.grad(output, hidd, allow_unused=True)[0]
        features_grad = features_grad[0]
        np.save('./feature/'+out_layer, features_grad)

        grad1 = np.load('./feature/fc00.npy')
        grad2 = np.load('./feature/fc01.npy')
        grad1 = np.maximum(grad1, 0)
        grad2 = np.maximum(grad2, 0)
        width = 0.5  # the width of the bars
        x = np.array([a for a in range(512)])
        fig, ax = plt.subplots(figsize=(10,5))
        rects1 = ax.bar(x-width/2, grad1, width, label='h1')
        rects2 = ax.bar(x+width/2, grad2, width, label='h2')

        ax.set_ylabel('Scores')
        ax.set_title('The importance of h1 and h2')
        ax.legend()
        plt.savefig('./feature/feature', dpi=600)
        plt.show()

if __name__ == '__main__':
    fusion_or_not = False
    if fusion_or_not == False:
        # 指定层名
        out_layer_name = 'conv1'#"ica1.fc11"
        features_grad = 0

        # 构建模型并加载预训练参数
        model_name = ['pocket_only_net', 'ami_only_net', 'pocket_and_ami', 'ami_att_net2', 'pocket_att_net2_1.27_1.46_1.54_refined', 'ami_att_net'] #'pocket_att_net2_1.21_1.52_1.48_general'
        PATH = './model/saved_model/model_' + model_name[4] + '.model'
        device = 'cpu'
        model = pocket_att_net2().to(device)
        model.load_state_dict(torch.load(PATH))
    else:
        # 指定层名
        out_layer_name = 'fc01'
        PATH = './model/saved_model/model_att_fusion2_1.254_1.503_refined.model'
        checkpoint = torch.load(PATH)
        device = 'cpu'
        model1 = pocket_att_net2().to(device)
        model2 = ami_att_net2().to(device)
        model = att_fusion2().to(device)
        model1.load_state_dict(checkpoint['model1'])
        model2.load_state_dict(checkpoint['model2'])
        model.load_state_dict(checkpoint['model'])

    li_name2, prot_name2, affinity = [], [], []
    with open('./processed/data_list/input_list.txt') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            li_name2.append(line[4])
            prot_name2.append(line[2])
            affinity.append(line[5])
    D = None
    name = '1bzc'
    with open('split_idx_fast_new', 'r') as f:
        ind = f.read()
        idx = ast.literal_eval(ind)
        data_idx_test = np.array(idx[2])
    id = 2
    if name == None:
        li_name = li_name2[data_idx_test[id]]
        prot_name = prot_name2[data_idx_test[id]]
        affinity = affinity[data_idx_test[id]]
        pdb_id = li_name[-15:-11]
    else:
        pdb_id = name
        li_name = './raw_data/PDBbind_v2020_refined/refined-set/'+pdb_id+'/'+pdb_id+'_ligand.pdb'#'./raw_data/refined-set/' + name + '/' + name +'_ligand.pdb'
        prot_name = './raw_data/PDBbind_v2020_refined/refined-set/'+pdb_id+'/'+pdb_id+'_pocket_8.pdb'#'./raw_data/refined-set/' + name + '/' + name +'_pocket_6.pdb'
        affinity = '4.92'

    torchinfo.summary(model)
    #test_data = GraphPairDataset(dataset='test', loc='./processed/多batch版本/general_2013/')
    test_loader = get_data(pdb_id=[pdb_id], ligand=[li_name], prot=[prot_name], affi=[affinity])


    mol = Chem.MolFromPDBFile(li_name)
    smi = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(smi)
    for batch_idx, data in enumerate(test_loader):
        D = data
    draw_CAM(D, visual_heatmap=True, out_layer=out_layer_name, model=model, mol=mol)#, model1=model1, model2=model2)