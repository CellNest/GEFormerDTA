import sys
print("当前编码格式是：", sys.getdefaultencoding())
# from rdkit import Chem
# from rdkit.Chem import AllChem
# from rdkit.Chem import Draw
#
# molecula = open('./smile.txt')  # 下载得到的小分子smiles格式
# moleculas = molecula.readlines()
# smile = []
# for i in moleculas:
#     smile.append(i)
# print(smile)
# mol = AllChem.AddHs(Chem.MolFromSmiles(smile))
# AllChem.EmbedMolecule(mol)
# AllChem.MMFFOptimizeMolecule(mol)
# Chem.MolToMolFile(mol, './smile' + str(i) + '.sdf')

from rdkit import Chem
# from rdkit.Chem import Draw, AllChem
from rdkit.Chem import Descriptors
import os
import pandas as pd

# 想要输入的化合物smile分子式
df = pd.read_csv('kiba/split_4/kiba_valid_setting_1.csv')  # 下载得到的小分子smiles格式
smilesList = df['compound_iso_smiles']

# smilesList = [
#     'C1=CC=CC=C1[Cl]',
# ]
# 化合物mol列表
# mols = []
i=0
for smile in smilesList:
    mol = Chem.MolFromSmiles(smile)
    # mols.append(mol)
    i=i+1
# 获取此python脚本的工作路径，并设置输出的SDF文件位置在当前工作目录下的/files/batch.sdf
# work_patch = os.getcwd()
# writer = Chem.SDWriter(work_patch + '/files/batch.sdf')
    writer = Chem.SDWriter('kiba/sdf/'+smile+'.sdf')

    writer.SetProps(['LOGP', 'MW', 'TPSA', '价电子数'])  # 设置SDF文件包含的化合物性质
    # for i, mol in enumerate(mols):
    mw = Descriptors.ExactMolWt(mol)
    logP = Descriptors.MolLogP(mol)
    TPSA = Descriptors.TPSA(mol)
    ValueElectronsNum = Descriptors.NumValenceElectrons(mol)
    name = Descriptors.names
    mol.SetProp('MW', '%.2f' % mw)  # 设置分子量
    mol.SetProp('LOGP', '%.2f' % logP)  # 设置分子logP
    mol.SetProp('_Name', 'No_%s' % i)  # 设置分子别名
    print('第'+str(i)+'个smile序列输出')
    mol.SetProp('TPSA', '%s' % TPSA)  # 设置分子的拓扑极性面积
    # mol.SetProp('chem_name', '%s' % name)
    mol.SetProp('价电子数', '%s' % ValueElectronsNum)  # 设置分子的价电子总数
    writer.write(mol)
    # print(mol)
    writer.close() # 关闭写入文件
