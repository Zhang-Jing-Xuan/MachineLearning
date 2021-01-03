from math import log2
import pandas as pd
import Decision_Tree_Visual
# from matplotlib.font_manager import FontProperties

# 统计label出现次数
def get_counts(data):
    total = len(data)
    results = {}
    for d in data:
        results[d[-1]] = results.get(d[-1], 0) + 1
        # get()函数：如果不存在，返回0，否则返回d[-1]所对应的值（是，否）
    return results, total

# 计算信息熵
def calcu_entropy(data):
    results, total = get_counts(data)
    ent = sum([-1.0*v/total*log2(v/total) for v in results.values()])
    return ent

# 计算每个feature的信息增益
def calcu_each_gain(column, update_data):
    total = len(column)
    grouped = update_data.iloc[:, -1].groupby(by=column)
    temp = sum([len(g[1])/total*calcu_entropy(g[1]) for g in list(grouped)])
    return calcu_entropy(update_data.iloc[:, -1]) - temp

# 获取最大的信息增益的feature
def get_max_gain(temp_data):
    columns_entropy = [(col, calcu_each_gain(temp_data[col], temp_data)) for col in temp_data.iloc[:, :-1]]
    columns_entropy = sorted(columns_entropy, key=lambda f: f[1], reverse=True)
    return columns_entropy[0]

# 去掉数据中已存在的列属性内容
def drop_exist_feature(data, best_feature):
    attr = pd.unique(data[best_feature])
    new_data = [(nd, data[data[best_feature] == nd]) for nd in attr]
    new_data = [(n[0], n[1].drop([best_feature], axis=1)) for n in new_data]
    return new_data

# 获得出现最多的label
def get_most_label(label_list):
    label_dict = {}
    for l in label_list:
        label_dict[l] = label_dict.get(l, 0) + 1
    sorted_label = sorted(label_dict.items(), key=lambda ll: ll[1], reverse=True)
    return sorted_label[0][0]

# 创建决策树
def create_tree(data_set, column_count):
    label_list = data_set.iloc[:, -1]
    if len(pd.unique(label_list)) == 1:#D中样本全属于同一类别
        return label_list.values[0]
    if all([len(pd.unique(data_set[i])) ==1 for i in data_set.iloc[:, :-1].columns]):#D中样本在column上取值相同
        return get_most_label(label_list)
    best_attr = get_max_gain(data_set)[0]
    tree = {best_attr: {}}
    exist_attr = pd.unique(data_set[best_attr])
    if len(exist_attr) != len(column_count[best_attr]):
        no_exist_attr = set(column_count[best_attr]) - set(exist_attr)
        for nea in no_exist_attr:
            tree[best_attr][nea] = get_most_label(label_list)
    for item in drop_exist_feature(data_set, best_attr):
        tree[best_attr][item[0]] = create_tree(item[1], column_count)
    return tree

if __name__ == '__main__':
    my_data = pd.read_csv('/Users/admin/Desktop/CL/Python/data/2.0.csv',encoding='utf8')
    column_count = dict([(ds, list(pd.unique(my_data[ds]))) for ds in my_data.iloc[:, :-1].columns])
    # column_count={'色泽': ['青绿', '乌黑', '浅白'], '根蒂': ['蜷缩', '稍蜷', '硬挺'], '敲声': ['浊响', '沉闷', '清脆'], 
    # '纹理': ['清晰', '稍糊', '模糊'], '脐部': ['凹陷', '稍凹', '平坦'], 
    # '触感': ['硬滑', '软粘']}
    d_tree = create_tree(my_data, column_count)
    # d_tree={'纹理': {'清晰': {'根蒂': {'蜷缩': '是', '稍蜷': 
    # {'色泽': {'浅白': '是', '青绿': '是', 
    # '乌黑': {'触感': {'硬滑': '是', '软粘': '否'}}}}, '硬挺': '否'}}, 
    # '稍糊': {'触感': {'软粘': '是', '硬滑': '否'}}, '模糊': '否'}}
    Decision_Tree_Visual.createTree(d_tree,"ID3决策树_西瓜数据集2.0")


