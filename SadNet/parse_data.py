from rdkit import Chem
import scipy.sparse as sp
from rdkit.Chem import rdmolops
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import os
import csv
import json
import ast
import math
import pandas as pd
import numpy as np
import joblib

class Parser(object):
    def __init__(self, cutoff=5,code_name=[],li_name=[],li_name2=[],prot_name=[],prot_name2=[],affinity=[],normalize=False,shuffle=False,external_test_data='2020'):
        self.code_name = code_name
        self.prot_name = prot_name
        self.prot_name2 = prot_name2
        self.li_name = li_name
        self.li_name2 = li_name2
        self.affinity = affinity
        self.sh = shuffle
        self.use_core = False #用之前的划分
        self.external_test_data=external_test_data
        self.core_idx = []
        self.des_list = [x[0] for x in Descriptors._descList]

        self.cutoff = cutoff
        self.atom_type = {}
        for at in ['C','N','O','P','S','B','mental','halogen','unknown']:
            self.atom_type[at] = 0
        self.num_atoms = 0
        self.x_ligand, self.x_pocket, self.pdb_code = [], [], []
        self.exclude = []
        self.dataframe = pd.DataFrame()

        self.normalize = normalize
        self.mean, self.std = 0, 1
        self.num_atoms,self.num_drugs,self.num_prot = 0,0,0
        self.num_drugs_features = 0
        self.num_prot_features = 0

        self.data_idx = None
        try:
            with open('split_idx_refined_fast_del2013_6A','r') as f:
                id = f.read()
                idx = ast.literal_eval(id)
                self.data_idx_train = np.array(idx[0])
                self.data_idx_valid = np.array(idx[1])
                self.data_idx_test = np.array(idx[2])
                self.data_idx_2013 = None
                if len(idx)==4:
                    self.data_idx_2013 = np.array(idx[3])
                #self.sh=True
        except:
            self.data_idx_train = None
            self.data_idx_valid = None
            self.data_idx_test = None
            self.data_idx_2013 = None
            #self.sh = True
            print('重新shuffle')

        self.train, self.valid, self.test = None, None, None
        self.train_drugs_x, self.valid_drugs_x, self.test_drugs_x = None, None, None
        self.train_prot_x, self.valid_prot_x, self.test_prot_x = None, None, None
        self.train_y, self.valid_y, self.test_y = None, None, None
        self.train_drugs_intra, self.valid_drugs_intra, self.test_drugs_intra = None, None, None
        self.train_prot_intra, self.valid_prot_intra, self.test_prot_intra = None, None, None
        self.train_a_inter, self.valid_a_inter, self.test_a_inter = None, None, None
        self.train_step, self.valid_step, self.test_step = 0, 0, 0

        if self.external_test_data != None:
            #self.use_core = False
            if os.path.exists('./{}A_'.format(self.cutoff) + self.external_test_data +'.pkl'):
                print('数据存在，提取数据')
                # get x,a,y
                # self.parse_dataset()
                self.get_x_a_y()
                #self.split()

            else:
                self.parse_dataset()
                self.dataframe = pad_data(self.dataframe, num_drugs=79, num_prot=192)#177,281 #79,192;275;381
                # Save data
                self.save_dataset()

                # get x,a,y
                print('生成完毕，提取数据')
                self.get_x_a_y()
                #self.split()
        else:
            if os.path.exists('./{}A_refined.pkl'.format(self.cutoff)):
                print('数据存在，提取数据')
                self.get_x_a_y()
                self.split()

            else:
                self.parse_dataset()
                # Save data
                self.dataframe = pad_data(self.dataframe)#, num_drugs=79, num_prot=192)
                self.save_dataset()

                #get x,a,y
                print('生成完毕，提取数据')
                self.get_x_a_y()
                self.split()

    def parse_dataset(self):
        def _one_hot(x, allowable_set):
            return list(map(lambda s: x == s, allowable_set))

        hydrogen_donor = Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]")  # 匹配提供H原子的原子
        hydrogen_acceptor = Chem.MolFromSmarts(
            "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]")
        acidic = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")  # 酸性
        basic = Chem.MolFromSmarts(
            "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]")

        for j in range(len(self.code_name)):
            mol_li = Chem.MolFromPDBFile(self.li_name2[j])
            #mol_li = Chem.MolFromMol2File(self.li_name[j])
            #if mol_li == None:
            #    mol_li = Chem.MolFromMolFile(self.li_name2[j])
            mol_prot = Chem.MolFromPDBFile(self.prot_name2[j])
            #if mol_prot == None:
            #    mol_prot = Chem.MolFromMolFile(self.prot_name2[j])
            print('{}/{}'.format(j+1,len(self.code_name)))
            #print(len(self.exclude))
            if mol_li is None or mol_prot is None:
                self.exclude.append(self.code_name[j])
            else:
                mol = Chem.CombineMols(mol_li, mol_prot)
                for atom in mol.GetAtoms():
                    symbol = atom.GetSymbol()  # 元素种类

                    if symbol not in self.atom_type.keys():
                        if symbol in ['Br','Cl','F','I']:
                            self.atom_type['halogen'] += 1
                        if symbol in ['Zn','Mg','Ca','Na','Fe','Cu','Hg','Sn','Sb','K','V','Co','Ga','Ru','Re','Ni','Mn','Cd']:
                            self.atom_type['mental'] += 1
                        else:
                            self.atom_type['unknown'] += 1
                    else:
                        self.atom_type[symbol] += 1
        #with open('./processed/exclude',mode='w') as file:
        #    for i in self.exclude:
        #        file.write(str(i)+',')
        print(self.atom_type)
        self.atom_type = {k: v for k, v in sorted(self.atom_type.items(), key=lambda item: item[1], reverse=True)}
        print(self.atom_type)
        print('{}个文件为空打不开，丢弃'.format(len(self.exclude)))

        columns = ['code', 'symbol', 'atomic_num', 'degree', 'hybridization', 'implicit_valence', 'formal_charge',
                   'aromaticity', 'ring_size', 'num_hs', 'acid_base', 'h_donor_acceptor', 'adjacency_intra_drug',
                   'adjacency_intra_prot','adjacency_ami', 'output','symbol_p', 'atomic_num_p', 'degree_p', 'hybridization_p', 'implicit_valence_p', 'formal_charge_p',
                   'aromaticity_p', 'ring_size_p', 'num_hs_p', 'acid_base_p', 'h_donor_acceptor_p','prot_emb','distance_ami','distance_li','edge_feat_l','edge_feat_p','inter']#,'ligand_or_prot','center_distance']
        self.dataframe = pd.DataFrame(columns=columns)

        for j in range(len(self.code_name)):
            xxx,yyy,zzz = 0,0,0
            if self.code_name[j] in self.exclude:
                continue
            mol_li = Chem.MolFromPDBFile(self.li_name2[j])
            #mol_li = Chem.MolFromMol2File(self.li_name[j])
            #if mol_li == None:
            #    mol_li = Chem.MolFromMolFile(self.li_name2[j])
            mol_prot = Chem.MolFromPDBFile(self.prot_name2[j])
            #if mol_prot == None:
            #    mol_prot = Chem.MolFromMolFile(self.prot_name2[j])
            mol = Chem.CombineMols(mol_li, mol_prot)
            m = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],[], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
            m[0] = self.code_name[j]
            num_prot = mol_prot.GetNumAtoms()
            num_ligand = mol_li.GetNumAtoms()
            adjacency = np.array(rdmolops.Get3DDistanceMatrix(mol))  # 返回分子的3D距离矩阵
            idx = np.argwhere(  # argwhere是返回满足条件的索引
                np.min(adjacency[:, :num_ligand], axis=1) <= self.cutoff).flatten().tolist() # 挑选出需要计算的原子序号
            m[32] = adjacency[:, idx][idx, :][:num_ligand, num_ligand:]
            idx_p = [x-num_ligand for x in idx]
            idx_p = [x for x in idx_p if x>=0]
            if len(idx_p) > 192:
                print(self.code_name[j] + '数量太多')
                continue
            #inter = [[], [], []]
            for k in range(num_ligand):
                x0 = mol_li.GetConformer().GetAtomPosition(k).x
                y0 = mol_li.GetConformer().GetAtomPosition(k).y
                z0 = mol_li.GetConformer().GetAtomPosition(k).z
                xxx += x0
                yyy += y0
                zzz += z0

            xxx, yyy, zzz = xxx / num_ligand, yyy / num_ligand, zzz / num_ligand

            m[12] = np.array(rdmolops.GetAdjacencyMatrix(mol_li))  # 配体内部原子邻接关系矩阵
            I_li = np.matrix(np.eye(m[12].shape[0]))
            m[12] = m[12] + I_li
            tmp_coo = sp.coo_matrix(m[12])
            drug_coo = np.vstack((tmp_coo.row, tmp_coo.col))
            m[12] = drug_coo.tolist()
            edge_feat_l = []
            for i in range(len(tmp_coo.row)):
                idx1 = int(tmp_coo.row[i])
                idx2 = int(tmp_coo.col[i])
                bond = mol.GetBondBetweenAtoms(idx1, idx2)
                if bond == None:
                    edge_feat_l.append([0, 0, 0, 0, adjacency[idx1, idx2]])
                    continue
                tmp = []
                tmp.append(bond.GetBondTypeAsDouble())
                if bond.GetIsAromatic() == True:
                    tmp.append(1)
                else:
                    tmp.append(0)
                if bond.GetIsConjugated() == True:
                    tmp.append(1)
                else:
                    tmp.append(0)
                if bond.IsInRing() == True:
                    tmp.append(1)
                else:
                    tmp.append(0)
                tmp.append(adjacency[idx1, idx2])
                edge_feat_l.append(tmp)
            m[30] = np.array(edge_feat_l)

            prot_adj = np.array(rdmolops.GetAdjacencyMatrix(mol_prot))
            I_pro = np.matrix(np.eye(prot_adj.shape[0]))
            m[13] = prot_adj + I_pro
            tmp_coo2 = sp.coo_matrix(m[13])
            prot_coo = np.vstack((tmp_coo2.row, tmp_coo2.col))
            m[13] = prot_coo.tolist()
            edge_feat_p = []
            for i in range(len(tmp_coo2.row)):
                idx1 = int(tmp_coo2.row[i])
                idx2 = int(tmp_coo2.col[i])
                if idx1 in idx_p and idx2 in idx_p:                     #剔除5A以外的口袋原子邻接关系
                    bond = mol.GetBondBetweenAtoms(idx1, idx2)
                    if bond == None:
                        edge_feat_p.append([0, 0, 0, 0, adjacency[idx1, idx2]])
                        continue
                    tmp = []
                    tmp.append(bond.GetBondTypeAsDouble())
                    if bond.GetIsAromatic() == True:
                        tmp.append(1)
                    else:
                        tmp.append(0)
                    if bond.GetIsConjugated() == True:
                        tmp.append(1)
                    else:
                        tmp.append(0)
                    if bond.IsInRing() == True:
                        tmp.append(1)
                    else:
                        tmp.append(0)
                    tmp.append(adjacency[idx1, idx2])
                    edge_feat_p.append(tmp)
            m[31] = np.array(edge_feat_p)
            m[13] = prot_adj + I_pro
            m[13] = m[13][idx_p, :][:, idx_p]     #剔除5A以外的口袋原子
            tmp_coo2 = sp.coo_matrix(m[13])
            prot_coo = np.vstack((tmp_coo2.row, tmp_coo2.col))
            m[13] = prot_coo.tolist()
            '''交互作用矩阵
            adj = np.zeros_like(aa)  # 形状一致的0矩阵
            I = np.matrix(np.eye(aa.shape[0]))
            for ind in idx:
                if ind > num_ligand-1:
                    adj[:num_ligand, ind] = 1.  # [0……1]
                    adj[ind, :num_ligand] = 1.  # [ …… ]
            m[14] = adj + I  # [1……0]    配体原子与口袋原子邻接关系矩阵
            '''
            try:
                my_npz = np.load('./processed/protein_embed_2016_general.npz', allow_pickle=True)
                emb = my_npz[self.code_name[j]].reshape(-1, 1)[0][0]['seq'][1:-1, :]
            except:
                my_npz = np.load('./processed/protein_embed_2020_refined.npz', allow_pickle=True)
                emb = my_npz[self.code_name[j]].reshape(-1, 1)[0][0]['seq'][1:-1, :]
            if len(emb) > 1000:
                emb = emb[:1000, :]
            m[27] = emb
            print(self.code_name[j])
            #提取阿尔法C原子坐标

            try:
                path = './processed/protein_αC_general/'+self.code_name[j]+'_protein_c.pdb'
                file = open(path, 'r')
            except:
                path = './processed/protein_αC_2020_refined/'+self.code_name[j]+'_protein_c.pdb'
                file = open(path, 'r')
            x,y,z,ami_map,all_map,distance, distance_li = [],[],[],[],[],[],[]
            lines = file.readlines()
            for line in lines:
                li = line.split()
                if li[0] == 'ATOM' and int(li[1]) <= 1000:
                    try:
                        if len(li[5].split('.', 2)[1]) == 3:
                            x.append(float(li[5]))
                            y.append(float(li[6]))
                            z.append(float(li[7]))
                    except BaseException:
                        try:
                            x.append(float(li[6]))
                        except BaseException:
                            if len(li[6]) > 20:
                                aaa = li[6].split('.', 4)
                                aaa1 = aaa[0] + '.' + aaa[1][:3]
                                aaa2 = aaa[1][3:] + '.' + aaa[2][:3]
                                aaa3 = aaa[2][3:] + '.' + aaa[3]
                                x.append(float(aaa1))
                                y.append(float(aaa2))
                                z.append(float(aaa3))
                                continue
                            else:
                                aaa = li[6].split('.', 3)
                                aaa1 = aaa[0] + '.' + aaa[1][:3]
                                aaa2 = aaa[1][3:] + '.' + aaa[2]
                                x.append(float(aaa1))
                                y.append(float(aaa2))
                                z.append(float(li[7]))
                                continue
                        try:
                            y.append(float(li[7]))
                        except BaseException:
                            aaa = li[7].split('.', 3)
                            aaa1 = aaa[0] + '.' + aaa[1][:3]
                            aaa2 = aaa[1][3:] + '.' + aaa[2]
                            y.append(float(aaa1))
                            z.append(float(aaa2))
                            continue
                        z.append(float(li[8]))

            for i in range(len(emb)):
                try:
                    dis_li = pow(pow(x[i] - xxx, 2) + pow(y[i] - yyy, 2) + pow(z[i] - zzz, 2), 0.5)
                    distance_li.append(dis_li)
                except:
                    distance_li.append(0)

            i = 0
            print(len(emb))
            l = min(len(emb),len(x))
            while i < l:
                jj = 0
                while jj < l:
                    dis =pow(pow(x[i] - x[jj], 2) + pow(y[i] - y[jj], 2) + pow(z[i] - z[jj], 2), 0.5)
                    if dis < 8:
                        ami_map.append([i,jj])
                        distance.append(1/(dis+1))
                    jj = jj + 1
                i = i + 1
            m[14] = ami_map
            m[28] = distance
            std = np.std(distance_li)
            mean = np.mean(distance_li)
            m[29] = -(distance_li - mean)/std

            #descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(self.des_list)
            #descriptors = descriptor_calculator.CalcDescriptors(mol_li)
            #m[32] = descriptors

            #m[16]
            D = np.array(rdmolops.Get3DDistanceMatrix(mol))  # cutoff内的原子距离矩阵
            #m[14] = np.where(m[16]<self.cutoff,m[14],0)
            m[15] = float(self.affinity[j])

            mol = [mol_li,mol_prot]
            for mol in mol:
                if mol==mol_li:
                    index = 0
                    idn = list(range(num_ligand))
                else:
                    index = 15
                    idn = idx_p

                Chem.AssignStereochemistry(mol)
                hydrogen_donor_match = sum(mol.GetSubstructMatches(hydrogen_donor), ())
                hydrogen_acceptor_match = sum(mol.GetSubstructMatches(hydrogen_acceptor), ())
                acidic_match = sum(mol.GetSubstructMatches(acidic), ())
                basic_match = sum(mol.GetSubstructMatches(basic), ())
                ring = mol.GetRingInfo()                                                            # 获取环的信息

                for atom_idx in idn:
                    atom = mol.GetAtomWithIdx(atom_idx)                                             # 通过索引获得原子
                    m[index+1].append(_one_hot(atom.GetSymbol(), self.atom_type.keys()))                  # 获得原子类型的矩阵
                    m[index+2].append([atom.GetAtomicNum()])                                              # 原子序号
                    m[index+3].append(_one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]))                  # 度矩阵
                    m[index+4].append(_one_hot(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,  # 返回原子杂化类型
                                                                   Chem.rdchem.HybridizationType.SP2,
                                                                   Chem.rdchem.HybridizationType.SP3,
                                                                   Chem.rdchem.HybridizationType.SP3D,
                                                                   Chem.rdchem.HybridizationType.SP3D2]))
                    m[index+5].append(_one_hot(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]))         # 返回原子隐式价态
                    m[index+6].append(_one_hot(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3]))         # 返回电荷数
                    m[index+7].append([atom.GetIsAromatic()])                                             # 判断原子是否是芳香性原子
                    m[index+8].append([ring.IsAtomInRingOfSize(atom_idx, 3),                              # 查看原子是否在n元环中
                                 ring.IsAtomInRingOfSize(atom_idx, 4),
                                 ring.IsAtomInRingOfSize(atom_idx, 5),
                                 ring.IsAtomInRingOfSize(atom_idx, 6),
                                 ring.IsAtomInRingOfSize(atom_idx, 7),
                                 ring.IsAtomInRingOfSize(atom_idx, 8)])
                    m[index+9].append(_one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]))                    # 返回这个原子上有几个H
                    m[index+10].append([atom_idx in acidic_match, atom_idx in basic_match])               # 提取出两种原子类别的序号
                    m[index+11].append([atom_idx in hydrogen_donor_match, atom_idx in hydrogen_acceptor_match])
            #inter[1] = list(range(len(inter[1])))
            #m[32] = inter
            #self.dataframe = self.dataframe.append(pd.DataFrame([m], columns=columns), ignore_index=True, sort=True)
            self.dataframe = pd.concat([self.dataframe, pd.DataFrame([m], columns=columns)], ignore_index=True)

    def get_x_a_y(self):
        self.exclude500_list=[]
        with open('./processed/exclude500.txt','r') as ex:
            line = ex.read()
            self.exclude500 = line.split(',')                      #exclude500

        if self.external_test_data!=None:
            self.dataframe = pd.read_pickle('./{}A_'.format(self.cutoff)+self.external_test_data+'.pkl')       #选择提取哪些数据
        else:
            self.dataframe = pd.read_pickle('./{}A_refined.pkl'.format(self.cutoff))

        #for i in range(len(self.dataframe)):
        #    if self.dataframe.iloc[i]['code'] in self.exclude500:
        #        self.exclude500_list.append(i)

        #self.dataframe = pad_data(self.dataframe)
        #self.dataframe.to_pickle('./' + str(self.cutoff) + 'A_general_910.pkl')
        # 取出核心集pdb号
        core = []
        file = open('./raw_data/core.txt')
        core_list = file.read()
        core_list = core_list.split(',')
        #print(core_list)
        #print(len(core_list))
        for i in core_list:
            core.append(i.strip())
        for i in range(len(self.dataframe)):
            if self.dataframe.iloc[i]['code'] in core:
                if self.dataframe.iloc[i]['code'] not in self.exclude500:
                    self.core_idx.append(i)

        #datafram进一步处理提取x，a,y
        self.num_drugs = len(self.dataframe.iloc[0]['symbol'])
        self.num_prot = len(self.dataframe.iloc[0]['symbol_p'])
        self.num_atoms = self.num_prot + self.num_drugs
        features_drugs = ['acid_base', 'aromaticity', 'atomic_num', 'degree', 'formal_charge', 'h_donor_acceptor',
                    'hybridization', 'implicit_valence', 'num_hs', 'ring_size', 'symbol']#, 'center_distance']
        features_prot = ['acid_base_p', 'aromaticity_p', 'atomic_num_p', 'degree_p', 'formal_charge_p', 'h_donor_acceptor_p',
                          'hybridization_p', 'implicit_valence_p', 'num_hs_p', 'ring_size_p', 'symbol_p']
        for f in features_drugs:
            self.num_drugs_features += len(self.dataframe.iloc[0][f][0])  # 把每个属性的具体每一项都拿出来作为一个特征
        for f in features_prot:
            self.num_prot_features += len(self.dataframe.iloc[0][f][0])
        print('{}A Dataset: Maximum {} atoms, {} features'.format(self.cutoff, self.num_drugs+self.num_prot, self.num_drugs_features+self.num_prot_features))

        # Construct tensors
        #self.drugs_x = np.zeros((len(self.dataframe), self.num_drugs, self.num_drugs_features),dtype=np.int8)  # （3479，79，52）
        self.drugs_x = []

        self.drugs_intra = np.zeros((len(self.dataframe), self.num_drugs, self.num_drugs),dtype=np.int8)
        #self.prot_x = np.zeros((len(self.dataframe), self.num_prot, self.num_prot_features),dtype=np.int8)  # （3479，280，56）
        self.prot_x = []

        self.prot_intra = np.zeros((len(self.dataframe), self.num_prot, self.num_prot), dtype=np.int8)
        #self.drug_feature = np.zeros((len(self.dataframe), 208),dtype=np.float)
        #self.a_inter = np.zeros(len(self.dataframe),dtype=np.int8)
        for idx in range(len(self.dataframe)):
            #self.drugs_x[idx] = np.concatenate([np.array(self.dataframe.iloc[idx][k], dtype=int) for k in features_drugs], axis=1)  # 把每一个复合物的特征拼接起来
            #self.prot_x[idx] = np.concatenate([np.array(self.dataframe.iloc[idx][k], dtype=float) for k in features_prot], axis=1)  # 把每一个口袋的特征拼接起来

            #self.prot_intra[idx] = np.array(self.dataframe.iloc[idx]['adjacency_intra_prot'], dtype=int)
            #self.drug_feature[idx] = np.array(self.dataframe.iloc[idx]['drug_feature'],dtype=float)

            self.drugs_x.append(np.concatenate([np.array(self.dataframe.iloc[idx][k], dtype=int) for k in features_drugs], axis=1))
            self.prot_x.append(np.concatenate([np.array(self.dataframe.iloc[idx][k], dtype=float) for k in features_prot], axis=1))
        self.drugs_x = np.array(self.drugs_x)
        self.prot_x = np.array(self.prot_x)

        #self.drug_feature = self.drug_feature / self.drug_feature.max(axis=0)
        #self.drug_feature = np.nan_to_num(self.drug_feature)

        self.ami_embedding = np.array(self.dataframe['prot_emb'])

        self.ami_index = np.array(self.dataframe['adjacency_ami'])

        self.ami_dis = np.array(self.dataframe['distance_ami'])

        self.ami_dis_li = np.array(self.dataframe['distance_li'])

        #self.drug_feature = np.array(self.dataframe['drug_feature'])

        #self.a_inter = np.array(self.dataframe.iloc[:,3])
        self.y = self.dataframe['output'].to_numpy()

        self.edge_feat_l = np.array(self.dataframe['edge_feat_l'])
        self.edge_feat_p = np.array(self.dataframe['edge_feat_p'])

        self.drugs_intra = np.array(self.dataframe['adjacency_intra_drug'])
        self.prot_intra = np.array(self.dataframe['adjacency_intra_prot'])
        xxxx = self.dataframe['inter']
        self.inter = np.array(self.dataframe['inter'])

        print('Tensors drug_x: {},drug_intra: {},prot_x: {},prot_intra: {}, y: {}'.format(self.drugs_x.shape, self.drugs_intra.shape, self.prot_x.shape,
                                                                      self.prot_intra.shape,self.y.shape))


    def split(self):
        # Split dataset
        np.set_printoptions(threshold=1e6)
        if self.sh==True:
            left_list = np.append(self.data_idx_train, self.data_idx_valid)
            idx = np.random.permutation(left_list)
            self.data_idx_valid = idx[:736]
            self.data_idx_train = idx[736:]

            #self.data_idx_train = self.data_idx[:idx_valid]
            #self.data_idx_valid = self.data_idx[idx_valid:idx_test]
            #self.data_idx_test = self.data_idx[idx_test:]

        if self.use_core == True:
            self.data_idx_test = np.array(self.core_idx)
            left_list = []
            for i in range(len(self.y)):
                if i not in self.core_idx:
                    left_list.append(i)
            idx = np.random.permutation(left_list)
            self.data_idx_valid = idx[:375]#idx[:len(self.core_idx)]
            self.data_idx_train = idx[375:]#idx[len(self.core_idx):]

        #with open('split_idx_shuffle', 'w') as f:
        #    f.write('[{},{},{}]'.format(self.data_idx_train.tolist(),self.data_idx_valid.tolist(),self.data_idx_test.tolist()))

        self.train_drugs_x, self.valid_drugs_x, self.test_drugs_x = self.drugs_x[self.data_idx_train], \
                                                  self.drugs_x[self.data_idx_valid], \
                                                  self.drugs_x[self.data_idx_test]
        self.train_prot_x, self.valid_prot_x, self.test_prot_x = self.prot_x[self.data_idx_train], \
                                                                    self.prot_x[self.data_idx_valid], \
                                                                    self.prot_x[self.data_idx_test]
        self.train_y, self.valid_y, self.test_y = self.y[self.data_idx_train], \
                                                  self.y[self.data_idx_valid], \
                                                  self.y[self.data_idx_test]
        self.train_drugs_intra, self.valid_drugs_intra, self.test_drugs_intra = self.drugs_intra[self.data_idx_train], \
                                                                                self.drugs_intra[self.data_idx_valid], \
                                                                                self.drugs_intra[self.data_idx_test]
        self.train_prot_intra, self.valid_prot_intra, self.test_prot_intra = self.prot_intra[self.data_idx_train], \
                                                                             self.prot_intra[self.data_idx_valid], \
                                                                             self.prot_intra[self.data_idx_test]
        self.train_ami_index, self.valid_ami_index, self.test_ami_index = self.ami_index[self.data_idx_train],\
                                                                          self.ami_index[self.data_idx_valid],\
                                                                          self.ami_index[self.data_idx_test]
        self.train_ami_embedding, self.valid_ami_embedding, self.test_ami_embedding = self.ami_embedding[self.data_idx_train], \
                                                                                      self.ami_embedding[self.data_idx_valid], \
                                                                                      self.ami_embedding[self.data_idx_test]

        self.train_ami_dis, self.valid_ami_dis, self.test_ami_dis = self.ami_dis[self.data_idx_train], \
                                                                    self.ami_dis[self.data_idx_valid], \
                                                                    self.ami_dis[self.data_idx_test]
        self.train_ami_dis_li,self.valid_ami_dis_li,self.test_ami_dis_li = self.ami_dis_li[self.data_idx_train], \
                                                                           self.ami_dis_li[self.data_idx_valid], \
                                                                           self.ami_dis_li[self.data_idx_test]
        self.train_edge_feat_l, self.valid_edge_feat_l, self.test_edge_feat_l = self.edge_feat_l[self.data_idx_train],\
                                                                                self.edge_feat_l[self.data_idx_valid],\
                                                                                self.edge_feat_l[self.data_idx_test]
        self.train_edge_feat_p, self.valid_edge_feat_p, self.test_edge_feat_p = self.edge_feat_p[self.data_idx_train], \
                                                                                self.edge_feat_p[self.data_idx_valid], \
                                                                                self.edge_feat_p[self.data_idx_test]
        self.train_inter, self.valid_inter, self.test_inter = self.inter[self.data_idx_train], \
                                                              self.inter[self.data_idx_valid], \
                                                              self.inter[self.data_idx_test]
        '''
        self.drugs_x_2013, self.prot_x_2013, self.y_2013, self.drugs_intra_2013, self.prot_intra_2013, self.ami_index_2013,\
            self.ami_embedding_2013, self.ami_dis_2013, self.ami_dis_li_2013, self.edge_feat_l_2013, self.edge_feat_p_2013, self.inter_2013 = \
                self.drugs_x[self.data_idx_2013], self.prot_x[self.data_idx_2013], self.y[self.data_idx_2013], self.drugs_intra[self.data_idx_2013], \
                self.prot_intra[self.data_idx_2013], self.ami_index[self.data_idx_2013], self.ami_embedding[self.data_idx_2013], self.ami_dis[self.data_idx_2013], \
                self.ami_dis_li[self.data_idx_2013], self.edge_feat_l[self.data_idx_2013], self.edge_feat_p[self.data_idx_2013], self.inter[self.data_idx_2013]
        '''

    def save_dataset(self):
        #with open('./' + str(self.cutoff) + 'A_2016_general.pkl', 'wb') as fo:
        #    joblib.dump(self.dataframe, fo)
        name = 'A_refined.pkl'
        if self.external_test_data!=None:
            name = 'A_'+self.external_test_data+'.pkl'
        self.dataframe.to_pickle('./' + str(self.cutoff) + name)
        print('DataFrame saved')

        # Pad data
def pad_data(dataframe, num_drugs=0, num_prot=0):
    '''
    num_inter = 0
    for i in range(len(dataframe)):
        num_inter = max(len(dataframe.iloc[i]['inter'][0]), num_drugs)
    print(num_inter)
    for i in range(len(dataframe)):
        print('pad' + dataframe.iloc[i]['code'])
        delta = num_inter - len(dataframe.iat[i, 32]) # 每个样本的原子和最大值差了多少
        dataframe.iat[i, 32] = np.pad(dataframe.iat[i, 32], ((0, 0), (0, delta)), 'constant',constant_values=((0, 0), (0, 0)))
    return dataframe
    '''
    if num_drugs == 0:
        for i in range(len(dataframe)):
            # 在这里把每个样本的原子个数按最大值统一
            num_drugs = max(len(dataframe.iloc[i]['symbol']), num_drugs)
            num_prot = max(len(dataframe.iloc[i]['symbol_p']), num_prot)
            print('max prot number:{},max ligand number:{}'.format(num_prot, num_drugs))

    for i in range(len(dataframe)):
        print('pad' + dataframe.iloc[i]['code'])
        delta_drug = num_drugs - len(dataframe.iat[i, 1])
        delta_prot = num_prot - len(dataframe.iat[i, 16])  # 每个样本的原子和最大值差了多少

        for j in [1,3,4,5,6,7,8,9,10,11]:
            dataframe.iat[i, j] = np.pad(dataframe.iat[i, j], ((0, delta_drug), (0, 0)), 'constant',constant_values=((False, False), (False, False)))
        dataframe.iat[i, 2] = np.pad(dataframe.iat[i, 2], ((0, delta_drug), (0, 0)), 'constant',constant_values=((0, 0),(0, 0)))
        #dataframe.iat[i, 12] = np.pad(dataframe.iat[i, 12], ((0, 0),(0, delta_drug)), 'constant', constant_values=((0, 0),(0, 0)))

        for j in [16,18,19,20,21,22,23,24,25,26]:
            dataframe.iat[i, j] = np.pad(dataframe.iat[i, j], ((0, delta_prot), (0, 0)), 'constant',constant_values=((False, False), (False, False)))
        dataframe.iat[i, 17] = np.pad(dataframe.iat[i, 17], ((0, delta_prot), (0, 0)), 'constant',constant_values=((0, 0), (0, 0)))
        dataframe.iat[i, 32] = np.pad(dataframe.iat[i, 32], ((0, delta_drug),(0, delta_prot)), 'constant', constant_values=((float('inf'), float('inf')), (float('inf'), float('inf'))))
        #dataframe.iat[i, 13] = np.pad(dataframe.iat[i, 13], ((0, 0), (0, delta_prot)), 'constant',constant_values=((0, 0), (0, 0)))
    return dataframe

