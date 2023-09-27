# coding=UTF-8

import numpy as np
import pandas as pd
import openpyxl
import os  
import xlrd
import xlwt
from xlutils.copy import copy

from os.path import join, getsize 
def write_excel_xls(path, sheet_name, value):
    index = len(value)  
    workbook = xlwt.Workbook()  
    sheet = workbook.add_sheet(sheet_name)  
    for i in range(0, index):
        for j in range(0, len(value[i])):
            sheet.write(i, j, value[i][j])  
    workbook.save(path)  
    print("Done")
 
 
def write_excel_xls_append(path, value):
    index = len(value)  
    workbook = xlrd.open_workbook(path)  
    sheets = workbook.sheet_names()  
    worksheet = workbook.sheet_by_name(sheets[0])  
    rows_old = worksheet.nrows  
    new_workbook = copy(workbook)  
    new_worksheet = new_workbook.get_sheet(0)  
    for i in range(0, index):
        for j in range(0, len(value[i])):
            new_worksheet.write(i+rows_old, j, value[i][j])  
    new_workbook.save(path)  
    print("Appended")
 
 
def read_excel_xls(path):
    workbook = xlrd.open_workbook(path)  
    sheets = workbook.sheet_names()  
    worksheet = workbook.sheet_by_name(sheets[0])  
    for i in range(0, worksheet.nrows):
        for j in range(0, worksheet.ncols):
            print(worksheet.cell_value(i, j), "\t", end="")  
        print()
        
        
path = os.getcwd()
print ('path', path)
files= os.listdir(path) 
s = []
scalarNameList = 'Test_Accuracy_top1'
data_all = pd.DataFrame()

book_name_xls_cifar10 = 'Acc_cifar10_Adam_atan.xls'
sheet_name_xls_cifar10 = 'cifar10'
book_name_xls_cifar100 = 'Acc_cifar100_Adam_atan.xls'
sheet_name_xls_cifar100 = 'cifar100'
 
value_title = [["Model", "SettingAlpha", "SettingBeta", "TestAccuracy", "TrainingAccuracy", "TrainingTime"],]
#value_title = [["Model", "SettingAlpha", "SettingBeta", "TestAccuracy", "TrainingTime"],]
 
write_excel_xls(book_name_xls_cifar10, sheet_name_xls_cifar10, value_title)
write_excel_xls(book_name_xls_cifar100, sheet_name_xls_cifar100, value_title)

#write_excel_xls_append(book_name_xls, value2)
#read_excel_xls(book_name_xls)

for file in files: 
     if  '.txt' in file and '_Adam' in file and not os.path.isdir(file) and getsize(file)/float(1024)>22: #'ori' in file and
         with open(path+"/"+file,'r') as f:
             if '_c100' in file: #and '0.1' in file and '0.2' in file:
                 print()
                 data=''.join(f.readlines())
                 print ('file', file)
                 
                 model_name = file[0: file.index('_c100')]
                 setting_alpha = file[file.index('_Adam_')+7:file.index('b')]
                 if 'Mobile' not in file:
                     setting_alpha = file[file.index('_Adam_')+7:file.index('b')]
                     setting_beta = file[file.index('b')+1:file.index('.txt')]
                 else:
                     id1 = [i for i,x in enumerate(file) if x=='b']
                     setting_alpha = file[file.index('_Adam_')+7:id1[1]]
                     setting_beta = file[id1[1]+1:file.index('.txt')]  
                 if 'ori' in file:
                     setting_alpha = 0
                     setting_beta = 0
                 #print (model_name, 'cifar100' ) 
                 test_acc = data[data.index('test_top1_acc:')+14:data.index(', test_min_loss')]
                 train_acc = data[data.index('train_acc_top1:')+15:data.index(', train_min_loss')]
                 train_time = data[data.index('train time:')+12:data.index('\ntset time:')]
                 #print (test_acc) 
                 #row = [[model_name, setting_alpha, setting_beta, test_acc, train_acc, train_time]]
                 row = [[model_name, setting_alpha, setting_beta, test_acc, train_acc, train_time]]
                 print('row', row)
                 write_excel_xls_append(book_name_xls_cifar100, row)
#                 data=''.join(f.readlines())
#                 #print ('file', file)
#                 
#                 model_name = file[0: file.index('_c10')]
#                 setting_alpha = file[file.index('_A')+2:file.index('b')]
#                 if 'Mobile' not in file:
#                     setting_alpha = file[file.index('_A')+2:file.index('b')]
#                     setting_beta = file[file.index('b')+1:file.index('.txt')]
#                 else:
#                     id1 = [i for i,x in enumerate(file) if x=='b']
#                     setting_alpha = file[file.index('_A')+2:id1[1]]
#                     setting_beta = file[id1[1]+1:file.index('.txt')]  
#                 if 'ori' in file:
#                     setting_alpha = 0
#                     setting_beta = 0
#                 #print (model_name, 'cifar100' ) 
#                 test_acc = data[data.index('test_top1_acc:')+14:data.index(', test_min_loss')]
#                 train_acc = data[data.index('train_acc_top1:')+15:data.index(', train_min_loss')]
#                 train_time = data[data.index('train time:')+12:data.index('\ntset time:')]
#                 #print (test_acc) 
#                 row = [[model_name, setting_alpha, setting_beta, test_acc, train_acc, train_time]]
#                 print('row', row)
#                 write_excel_xls_append(book_name_xls_cifar100, row)
             
             elif '_c10' in file:
                 print()
                 data=''.join(f.readlines())
                 print ('file', file)
                 
                 model_name = file[0: file.index('_c10')]
                 setting_alpha = file[file.index('_Adam_')+7:file.index('b')]
                 if 'Mobile' not in file:
                     setting_alpha = file[file.index('_Adam_')+7:file.index('b')]
                     setting_beta = file[file.index('b')+1:file.index('.txt')]
                 else:
                     id1 = [i for i,x in enumerate(file) if x=='b']
                     setting_alpha = file[file.index('_Adam_')+7:id1[1]]
                     setting_beta = file[id1[1]+1:file.index('.txt')]  
                 if 'ori' in file:
                     setting_alpha = 0
                     setting_beta = 0
                 #print (model_name, 'cifar100' ) 
                 test_acc = data[data.index('test_top1_acc:')+14:data.index(', test_min_loss')]
                 train_acc = data[data.index('train_acc_top1:')+15:data.index(', train_min_loss')]
                 train_time = data[data.index('train time:')+12:data.index('\ntset time:')]
                 #print (test_acc) 
                 #row = [[model_name, setting_alpha, setting_beta, test_acc, train_acc, train_time]]
                 row = [[model_name, setting_alpha, setting_beta, test_acc, train_acc, train_time]]
                 print('row', row)
                 write_excel_xls_append(book_name_xls_cifar10, row)

             
         



 
 

 
 

 
 



         
#          f = open(path+"/"+file); 
#          iter_f = iter(f); 
#          str = ""
#          for line in iter_f: 
#              str = str + line
#          s.append(str) 
#
#
#with open('./VGG16_c100_SGD_ori_A0.1b20.txt','r') as f:
#    data=''.join(f.readlines())    


