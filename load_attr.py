
import pandas as pd
import numpy as np


def load_attr(path,num,pt,pv):

#traininig attr
    df_attr = pd.read_csv(path + 'list_attr_celeba.csv')
    df_attr.set_index('image_id', inplace=True)
    df_attr.replace(to_replace=-1, value=0, inplace=True)
    df_attr = df_attr[0:num]

    df_partition = pd.read_csv(path + 'list_eval_partition.csv')
    df_partition = df_partition[0:num]

    n_train = np.int(num * pt)
    n_val = np.int(num * pv) + n_train


    df_partition['partition'][0:n_train] = 0  # train
    df_partition['partition'][(n_train+1):n_val] = 1 #valiadation
    df_partition['partition'][(n_val+1):num] = 2 #test
    df_partition.set_index('image_id', inplace=True)
    df_par_attr = df_partition.join(df_attr['Male'], how='inner')

    #test attribute
    df_test = pd.read_csv(path + 'list_test_images.csv')
    df_test['image_id'] += '.jpg'
    df_test.set_index('image_id', inplace=True)
    df_par_attr = df_par_attr.append(df_test)

    return df_par_attr