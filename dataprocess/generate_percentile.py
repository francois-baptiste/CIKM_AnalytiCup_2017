import numpy as np
import pandas as pd

# produces percentile data
def percentile(line, data_type):

    cate = line[0].split(',')
    id_label = [cate[0]]

    if data_type == 'train':
        id_label.append(float(cate[1]))

    record = [int(cate[2])]
    length = len(line)

    for i in range(1,length):
        record.append(int(line[i]))

    mat = np.array(record).reshape(15,4,101,101)

    # deals with -1
    mat[mat == -1] = 0

    con_mat = np.zeros((15, 4, 10, 4))
    for i in range(15):

        for j in range(4):

            temp_mat = mat[i,j]

            for m in range(1,11):

                mt = temp_mat[50-5*m:50+5*m+1, 50-5*m:50+5*m+1]
                con_mat[i, j, m-1, 0] = np.max(mt)
                con_mat[i, j, m-1, 1] = np.percentile(mt,75,interpolation='lower')
                con_mat[i, j, m-1, 2] = np.percentile(mt,50,interpolation='lower')
                con_mat[i, j, m-1, 3] = np.percentile(mt,25,interpolation='lower')

    return id_label, con_mat.reshape(15*4*10*4)

# produces percentile data set
def data_process(filename, data_type):

    header_list = ['id']
    for i in range(15*4*10*4):
        feature = 'thxy_' + str(i+1)
        header_list.append(feature)

    if data_type == 'train':
        header_list += ['label']

    df = pd.DataFrame(columns=header_list)

    with open(filename) as fr:
        for line in fr:
            id_label, con_mat = percentile(line.strip().split(' '), data_type)
            simp = list(con_mat)
            temp = [id_label[0]]
            temp += simp

            if data_type == 'train':
                temp += [id_label[1]]

            print(temp)
            df_temp = pd.DataFrame([temp], columns=header_list)
            df = df.append(df_temp, ignore_index=True)
    return df


if __name__ == "__main__":
    trainfile = '../data/train.txt'
    testBfile = '../data/testB.txt'

    train_df = data_process(trainfile, data_type='train')
    test_df = data_process(testBfile, data_type='testB')
