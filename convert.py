#!/usr/bin/python
# coding=utf8

import sys
import re
import codecs
import os
from xml.etree.ElementTree import ElementTree as etree
from xml.etree.ElementTree import Element, SubElement, ElementTree

# 传入至少5个参数
# 生成译文的xml
# param 1: 要处理的是什么文件，只允许传入 "src","tst","ref"
# param 2：评测集名称
# param 3：源语言
# param 4：目标语言
# param 5..：要处理的文件

# 例如
# 生成原文
# python convert.py src xml_data/test Chinese English bleu_data/src.txt
# 生成译文
# python convert.py tst xml_data/test Chinese English bleu_data/tst.txt
# 生成ref
# python convert.py ref xml_data/test Chinese English bleu_data/ref.txt

# perl mteval-v13a.pl -r ../xml_data/test_ref.xml -s ../xml_data/test_src.xml -t ../xml_data/test_tst.xml


'''
@ 生成ref的xml
@ param 1: 多份ref的list，list中每一个元素为一个ref的list
@ param 2：评测集名称
@ param 3：源语言
@ param 3：目标语言
'''


def genrefxml(reflists, setid, srclang, trglang):
    mteval = Element('mteval')
    for reflist in reflists:
        sysid = reflist[0]
        set = SubElement(mteval, "refset")
        set.attrib = {"setid": setid, "srclang": srclang, "trglang": trglang, "refid": sysid}
        doc = SubElement(set, "doc")
        doc.attrib = {"docid": "doc1"}

        i = 0
        for sentence in reflist:
            # 第一位存储具体是哪个引擎
            if i != 0:
                p = SubElement(doc, "p")
                seg = SubElement(p, "seg")
                seg.attrib = {"id": str(i)}
                seg.text = sentence
            i = i + 1
    tree = ElementTree(mteval)
    tree.write(setid + '_ref.xml', encoding='utf-8')


'''
@ 生成译文的xml
@ param 1: 多份译文的list，list中每一个元素为一个译文的list
@ param 2：评测集名称
@ param 3：源语言
@ param 3：目标语言
'''


def gentstxml(tstlists, setid, srclang, trglang):
    mteval = Element('mteval')
    for tstlist in tstlists:
        sysid = tstlist[0]
        set = SubElement(mteval, "tstset")
        set.attrib = {"setid": setid, "srclang": srclang, "trglang": trglang, "sysid": sysid}
        doc = SubElement(set, "doc")
        doc.attrib = {"docid": "doc1"}

        i = 0
        for sentence in tstlist:
            # 第一位存储具体是哪个引擎
            if i != 0:
                p = SubElement(doc, "p")
                seg = SubElement(p, "seg")
                seg.attrib = {"id": str(i)}
                seg.text = sentence
            i = i + 1
    tree = ElementTree(mteval)
    tree.write(setid + '_tst.xml', encoding='utf-8')


'''
@ 生成原文的xml
@ param 1: 原文内容的list
@ param 2：评测集名称
@ param 3：源语言
'''


def gensrcxml(senlist, setid, srclang):
    mteval = Element('mteval')
    set = SubElement(mteval, "srcset")
    set.attrib = {"setid": setid, "srclang": srclang}
    doc = SubElement(set, "doc")
    doc.attrib = {"docid": "doc1"}

    i = 1
    for sentence in senlist:
        p = SubElement(doc, "p")
        seg = SubElement(p, "seg")
        seg.attrib = {"id": str(i)}
        seg.text = sentence
        i += 1
    tree = ElementTree(mteval)
    tree.write(setid + '_src.xml', encoding='utf-8')


# 调用具体的生成xml
def genxmltree(filetype, setid, srclang, trglang, files):
    if filetype not in ["src", "tst", "ref"]:
        print("filetype is error")
        return

    # 处理原文
    if filetype == "src":
        srclist = []
        for line in open(files[0]):
            line = line.strip()
            if line:
                srclist.append(line)
        gensrcxml(srclist, setid, srclang)

    # 处理译文
    if filetype == "tst":
        tstslist = []
        for tstfile in files:
            tstlist = []
            tstlist.append(str(tstfile).strip('.txt'))
            for line in open(tstfile):
                line = line.strip()
                if line:
                    tstlist.append(line)
            tstslist.append(tstlist)
        gentstxml(tstslist, setid, srclang, trglang)

    # 处理ref
    if filetype == "ref":
        reflists = []
        for reffile in files:
            reflist = []
            reflist.append(str(reffile).strip('.txt'))
            for line in open(reffile):
                line = line.strip()
                if line:
                    reflist.append(line)
            reflists.append(reflist)
        genrefxml(reflists, setid, srclang, trglang)


argv_len = len(sys.argv)
# print argv_len
if argv_len < 6:
    print("param error! src/ref tmq English Chinese 1.txt ")
    sys.exit()

filetype = sys.argv[1]
setid = sys.argv[2]
srclang = sys.argv[3]
trglang = sys.argv[4]
files = []
for i in range(5, len(sys.argv)):
    files.append(sys.argv[i])

genxmltree(filetype, setid, srclang, trglang, files)
