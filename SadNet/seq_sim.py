import os

test_1405 = []
with open('./raw_data/test1405.txt', 'r',encoding='utf-8') as f:
    for line in f.readlines():
        line = line.split()[0]
        if len(line) == 4:
            test_1405.append(line)
with open('./raw_data/2020_test_1405.txt', 'w') as f:
    for i in test_1405:
        f.write(str(i)+'\n')
print(test_1405)



test_2020_nosim = []
train_2016 = []
with open('./raw_data/train.txt', 'r') as f:
    for line in f.readlines():
        train_2016.append(line.strip())

c = 0
with open('./processed/DB_clu.tsv', 'r') as f:
    test_2020 = []
    cluster_key = None
    throw_out_flag = None
    for line in f.readlines():
        line = line.split()

        if line[0] != cluster_key:
            c += 1
            for i in test_2020:
                test_2020_nosim.append(i)
            test_2020 = []
            throw_out_flag = False
            cluster_key = line[0]
            if line[1] in train_2016:
                throw_out_flag = True
                test_2020 = []
                continue
            else:
                test_2020.append(line[1])
        elif throw_out_flag == True:
            continue
        else:
            if line[1] in train_2016:
                throw_out_flag = True
                test_2020 = []
                continue
            test_2020.append(line[1])

print(test_2020_nosim)
with open('./raw_data/2020_test_nosim.txt', 'w') as f:
    for i in test_2020_nosim:
        f.write(str(i)+'\n')
