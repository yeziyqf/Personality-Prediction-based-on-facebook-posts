#encoding:utf-8


import csv
import sys
import pandas as pd
import re
import string

# Extract csv column to txt files.
# print(sys.getdefaultencoding())
# csv_file = "./mypersonality_final_classifiedByClass_onlyColumn.csv"
#
# def _removeNonAscii(s): return "".join(i for i in s if ord(i)<128)
#
# content=pd.read_csv(csv_file,encoding='cp1252')    # This is special character friendly!!
# i =0
# printable = set(string.printable)
# for row in content.values:
#     filter(lambda x: x in printable, row)
#     if i <= 2442:
#         file_title = './txt output/y{}.txt'.format(i)
#         filter(lambda x: x in printable, row)
#         # s = row.encode('ascii', errors='ignore')
#         row.tofile(file_title, sep=",", format="%s")
#     else:
#         file_title = './txt output/n{}.txt'.format(i-2443)
#         filter(lambda x: x in printable, row)
#         # s = row.encode('ascii', errors='ignore')
#         row.tofile(file_title, sep=",", format="%s")
#     i += 1

####################################################################################
# import glob
# import re
#
# # For y folder
# # 采用glob模块匹配出目标文件夹下的所有txt后缀名的文件
# txt_filenames = glob.glob(r'./txt output_final/y/*.txt')
# print( txt_filenames)
#
# # 将每个txt文件中的信息放在一行中，并存储到目标文件中
# fileout = open(r'./txt output_final/y/sum/y_all.txt','w+') # 注意：AllDiary.txt需要是已经创建过的文件
# for i in range(len(txt_filenames)):
#     txt_file = open(txt_filenames[i], 'r') # 可以转码成txt_filenames[i].decode('utf-8')，也可以不转码
#     buf = txt_file.read()  # the context of txt file saved to buf
#     content = buf.replace("\n", " ").strip()
#     p = content
#     fileout.write(p)
#     fileout.write('\n')
#     txt_file.close()
# fileout.close()
#
# # For n folder
# # 采用glob模块匹配出目标文件夹下的所有txt后缀名的文件
# txt_filenames = glob.glob(r'./txt output_final/n/*.txt')
# print( txt_filenames)
#
# # 将每个txt文件中的信息放在一行中，并存储到目标文件中
# fileout = open(r'./txt output_final/n/sum/n_all.txt','w+') # 注意：AllDiary.txt需要是已经创建过的文件
# for i in range(len(txt_filenames)):
#     txt_file = open(txt_filenames[i], 'r') # 可以转码成txt_filenames[i].decode('utf-8')，也可以不转码
#     buf = txt_file.read()  # the context of txt file saved to buf
#     content = buf.replace("\n", " ").strip()
#     p = content
#     fileout.write(p)
#     fileout.write('\n')
#     txt_file.close()
# fileout.close()

####################################################################
# Change to lower cases
from itertools import chain
from glob import glob

file = open('./txt output_final/y/sum/y_all_train.txt', 'r')

lines = [line.lower() for line in file]
with open('./txt output_final/y/sum/y_all_train.txt', 'w') as out:
     out.writelines(sorted(lines))



