import os

def get_path_affi(affinity_path = None,filepath = None,out=None,exclude=False):
    #取得亲和度数据
    affinity=dict()
    try:
        file=open(affinity_path,'r')
    except FileNotFoundError:
        print("File is not found")
    else:
        lines=file.readlines()
        for line in lines:
            if line[0]!="#":
                li=line.split()
                affinity.update({li[0]:li[3]})

    #取得蛋白质、配体文件名以及对应的亲和度数据
    exclude_list = list()
    if exclude == True:
        exclude_dir = os.listdir("./raw_data/refined-set")
        for n in exclude_dir:
            if n!="index" and n!="readme":
                exclude_list.append(n)
    code_name=list()
    l_name=list()
    p_name=list()
    a=list()
    p_name2=list()

    i=0
    j=0

    for f in filepath:
        Dir=os.listdir(f)
        for n in Dir:
            if n!="index" and n!="readme" and n not in exclude_list:
                l = f + "/" + n + "/" + n + "_ligand.mol2" + "\t" + f + "/" + n + "/" + n + "_ligand.pdb"
                p = f + "/" + n + "/" + n + "_pocket.pdb" + "\t" + f + "/" + n + "/" + n + "_pocket_8.pdb"
                p2 = f + "/" + n + "/" + n + "_protein.pdb"
                code_name.append(n)
                l_name.append(l)
                p_name.append(p)
                a.append(affinity[n])
                p_name2.append(p2)
                i+=1
        print(i)

    #把每组数据写入文件
    while j<i:
        line_w=code_name[j] + "\t" + p_name[j]+"\t"+l_name[j]+"\t"+a[j]+"\t"+p_name2[j]+"\n"
        out.write(line_w)
        j+=1

def get_2020_affi(affinity_path = None,filepath = None,out=None,exclude=False):
    #首先读取222个PDBID
    test_2020 = []
    file_2020 = open('./raw_data/2020_test_1405.txt')
    for line in file_2020.readlines():
        pdbid = line.strip()
        test_2020.append(pdbid)

    #取得亲和度数据
    affinity=dict()
    try:
        file=open(affinity_path,'r')
    except FileNotFoundError:
        print("File is not found")
    else:
        lines=file.readlines()
        for line in lines:
            if line[0]!="#":
                li=line.split()
                if li[0] in test_2020:
                    affinity.update({li[0]:li[3]})

    #取得蛋白质、配体文件名以及对应的亲和度数据
    code_name=list()
    l_name=list()
    p_name=list()
    a=list()
    p_name2=list()

    i=0
    j=0

    for f in filepath:
        Dir=os.listdir(f)
        for k in test_2020:
            if k not in Dir:
                print(k)
        for n in Dir:
            if n!="index" and n!="readme" and n in test_2020:
                l = f + "/" + n + "/" + n + "_ligand.mol2" + "\t" + f + "/" + n + "/" + n + "_ligand.pdb"
                p = f + "/" + n + "/" + n + "_pocket.pdb" + "\t" + f + "/" + n + "/" + n + "_pocket_8.pdb"
                p2 = f + "/" + n + "/" + n + "_protein.pdb"
                code_name.append(n)
                l_name.append(l)
                p_name.append(p)
                a.append(affinity[n])
                p_name2.append(p2)
                i+=1
        print(i)

    #把每组数据写入文件
    while j<i:
        line_w=code_name[j] + "\t" + p_name[j]+"\t"+l_name[j]+"\t"+a[j]+"\t"+p_name2[j]+"\n"
        out.write(line_w)
        j+=1

def get_2020_affi2(affinity_path = None,filepath = None,out=None,exclude=False):
    test_2016, test_2020 = [], []
    file = open("./raw_data/refined-set/index/INDEX_general_PL_data.2016", 'r')
    lines = file.readlines()
    for line in lines:
        if line[0] != "#":
            li = line.split()
            test_2016.append(li[0])

    #取得亲和度数据
    affinity=dict()
    try:
        file=open(affinity_path,'r')
    except FileNotFoundError:
        print("File is not found")
    else:
        lines=file.readlines()
        for line in lines:
            if line[0]!="#":
                li=line.split()
                if li[0] in test_2016:
                    continue
                test_2020.append(li[0])
                affinity.update({li[0]:li[3]})

    #取得蛋白质、配体文件名以及对应的亲和度数据
    code_name=list()
    l_name=list()
    p_name=list()
    a=list()
    p_name2=list()

    i=0
    j=0

    for f in filepath:
        Dir=os.listdir(f)
        for k in test_2020:
            if k not in Dir:
                print(k)
        for n in Dir:
            if n!="index" and n!="readme" and n in test_2020:
                l = f + "/" + n + "/" + n + "_ligand.mol2" + "\t" + f + "/" + n + "/" + n + "_ligand.pdb"
                p = f + "/" + n + "/" + n + "_pocket.pdb" + "\t" + f + "/" + n + "/" + n + "_pocket_8.pdb"
                p2 = f + "/" + n + "/" + n + "_protein.pdb"
                code_name.append(n)
                l_name.append(l)
                p_name.append(p)
                a.append(affinity[n])
                p_name2.append(p2)
                i+=1
        print(i)

    #把每组数据写入文件
    while j<i:
        line_w=code_name[j] + "\t" + p_name[j]+"\t"+l_name[j]+"\t"+a[j]+"\t"+p_name2[j]+"\n"
        out.write(line_w)
        j+=1


def get_2013_affi(affinity_path = None,filepath = None,out=None):
    code_name, l_name, p_name, a, p_name2 = [], [], [], [], []
    exclude = []
    with open('./raw_data/test.txt') as file:
        for line in file.readlines():
            line = line.split()
            exclude.append(line[0])
    affi = {}
    with open(affinity_path) as f:
        for line in f.readlines():
            if line[0] != '#':
                line = line.split()
                if line[0] not in exclude:
                    affi.update({line[0]:line[3]})

    Dir = os.listdir(filepath)
    for n in Dir:
        if n != 'index' and n != 'README' and n in affi.keys():
            l = filepath + "/" + n + "/" + n + "_ligand.mol2" + "\t" + filepath + "/" + n + "/" + n + "_ligand.pdb"
            p = filepath + "/" + n + "/" + n + "_pocket.pdb" + "\t" + filepath + "/" + n + "/" + n + "_pocket_8.pdb"
            p2 = filepath + "/" + n + "/" + n + "_protein.pdb"
            code_name.append(n)
            l_name.append(l)
            p_name.append(p)
            a.append(affi[n])
            p_name2.append(p2)
    for j in range(len(code_name)):
        line_w=code_name[j] + "\t" + p_name[j]+"\t"+l_name[j]+"\t"+a[j]+"\t"+p_name2[j]+"\n"
        out.write(line_w)

'''
#创建2016一般集
get_path_affi(affinity_path = "./raw_data/refined-set/index/INDEX_general_PL_data.2016",
              filepath = ["./raw_data/refined-set"],
              out=open('./processed/data_list/input_list.txt','w'),
              exclude=False)

get_path_affi(affinity_path='./raw_data/refined-set/index/INDEX_general_PL_data.2016',
              filepath=['./raw_data/general-set-except-refined'],
              out=open('./processed/data_list/input_list_general_pdb.txt','a'),
              exclude=False)

#创建2020fast测试机名单
get_2013_affi(affinity_path = "./raw_data/casf2013/index/2013_core_data.lst",
              filepath = "./raw_data/casf2013/",
              out=open('./processed/data_list/input_list_2013_new.txt','w'))

get_2020_affi(affinity_path = "./raw_data/PDBbind_v2020_refined/refined-set/index/INDEX_general_PL_data.2020",
              filepath=["./raw_data/PDBbind_v2020_refined/refined-set/"],
              out=open('./processed/data_list/input_list_2020_fast.txt','w'))
'''
get_2020_affi(affinity_path = "./raw_data/PDBbind_v2020_refined/refined-set/index/INDEX_refined_data.2020",
              filepath=["./raw_data/PDBbind_v2020_refined/refined-set/"],
              out=open('./processed/data_list/input_list_2020_1405.txt','w'))






