#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 22:56:07 2020

@author: chengyu
"""

import xml.etree.ElementTree as ET 

def get_abstract_from_xml(wiki_xml_path):
    """
    input: file path
    """
    # create element tree object 
    tree = ET.parse(file_path) 
    # get root element 
    root = tree.getroot() 

    texts = []
    for a in root.findall('doc/abstract'):
        try:
            t = a.text.strip()
            if len(t)>0:
                texts.append(a.text)
        except:
            pass
            
    return texts

def write_to_txt(string_list,outfile):
    """
    input: list of string and outfile: file path
    """
    with open(outfile, 'w', encoding='utf8') as filehandle:
        for listitem in string_list:
            filehandle.write('%s\n' % listitem)
    return None

#%%
if __name__ =="__main__":
    
    file_path = '../../data/wikizh/zhwiki-latest-abstract-zh-cn1.xml'
    out_file_train = '../../data/wikizh/wiki_zh_train.txt'
    out_file_test = '../../data/wikizh/wiki_zh_test.txt'

    content = get_abstract_from_xml(file_path)

    split_num = int(len(content)*0.7)
    content_train = content[:split_num]
    content_test = content[split_num:]
    write_to_txt(content_train,out_file_train)
    write_to_txt(content_test,out_file_test)
    
    