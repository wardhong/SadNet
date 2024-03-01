import os
map_all = []
path = './processed/protein_αC'
Dir_c = os.listdir(path)
for pdb in Dir_c:
    print(pdb[:4])
    x,y,z,sis,map = [],[],[],[],[]
    pdb_path = path + '/' + pdb
    file = open(pdb_path, 'r')
    '''
    mol_prot = Chem.MolFromPDBFile(pdb_path)
    num = mol_prot.GetNumAtoms()
    i=0
    while i < num:
        x.append(mol_prot.GetConformer().GetAtomPosition(i).x)
        y.append(mol_prot.GetConformer().GetAtomPosition(i).y)
        z.append(mol_prot.GetConformer().GetAtomPosition(i).z)
    '''
    lines = file.readlines()
    for line in lines:
        li = line.split()
        if li[0]=='ATOM' and int(li[1])<=1000:
            try:
                if len(li[5].split('.',2)[1])==3:
                    x.append(float(li[5]))
                    y.append(float(li[6]))
                    z.append(float(li[7]))
            except BaseException:
                try:
                    x.append(float(li[6]))
                except BaseException:
                    if len(li[6])>20:
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
                    aaa = li[7].split('.',3)
                    aaa1 = aaa[0] + '.' + aaa[1][:3]
                    aaa2 = aaa[1][3:] + '.' + aaa[2]
                    y.append(float(aaa1))
                    z.append(float(aaa2))
                    continue
                z.append(float(li[8]))



    #i = 0
    #while i < len(x)-1:
    #    seq.append([i,i+1])
    #    sis.append(pow(pow(x[i]-x[i+1],2)+pow(y[i]-y[i+1],2)+pow(z[i]-z[i+1],2),0.5))
    #    i = i+1
    #print(sis)
    i = 0
    while i < len(x)-1:
        j = i + 1
        while j < len(x):
            if pow(pow(x[i]-x[j],2)+pow(y[i]-y[j],2)+pow(z[i]-z[j],2),0.5)<8:
                map.append([i,j])
            j = j + 1
        i = i + 1
    map_all.append(map)
print(map_all)



'''
            map_all = []
            path = './processed/protein_αC'
            Dir_c = os.listdir(path)
            for pdb in Dir_c:
                if pdb[:4] in self.exclude:
                    continue
                print(pdb[:4])
                x, y, z, ami_map = [], [], [], []
                pdb_path = path + '/' + pdb
                file = open(pdb_path, 'r')
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
                i = 0
                while i < len(x) - 1:
                    j = i + 1
                    while j < len(x):
                        if pow(pow(x[i] - x[j], 2) + pow(y[i] - y[j], 2) + pow(z[i] - z[j], 2), 0.5) < 8:
                            ami_map.append([i, j])
                        j = j + 1
                    i = i + 1
                map_all.append(ami_map)
            # print(map_all)
            m[15] = map_all
'''

