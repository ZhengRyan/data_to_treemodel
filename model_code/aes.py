#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: aes.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-06-24
'''

from binascii import b2a_hex, a2b_hex

import pandas as pd
from Crypto.Cipher import AES


class prpcrypt():
    def __init__(self, key):
        self.key = self.handle_length(key)
        self.mode = AES.MODE_CBC

    def handle_length(self, text):
        # 这里密钥key 长度必须为16（AES-128）、24（AES-192）、或32（AES-256）Bytes 长度.目前AES-128足够用
        length = 16
        count = len(text)
        if (count % length != 0):
            add = length - (count % length)
        else:
            add = 0
        text = text + ('\0' * add)
        return text.encode()

    # 加密函数【加密文本text必须为16的倍数！】，如果text不是16的倍数,那就补足16位
    def encrypt(self, text):
        text = self.handle_length(text)
        cryptor = AES.new(self.key, self.mode, self.key)
        ciphertext = cryptor.encrypt(text)
        # 因为AES加密时候得到的字符串不一定是ascii字符集的，输出到终端或者保存时候可能存在问题
        # 所以这里统一把加密后的字符串转化为16进制字符串
        return b2a_hex(ciphertext).decode()

    # 解密函数
    def decrypt(self, text):
        cryptor = AES.new(self.key, self.mode, self.key)
        plain_text = cryptor.decrypt(a2b_hex(text))
        # 去掉加密时填补的空格，使用strip()
        return plain_text.decode().rstrip('\0')


if __name__ == '__main__':
    pc = prpcrypt('ecloud')  # 初始化密钥,小于等于16位
    # e = pc.encrypt("229488a89b8ca750c277ed3ff7342f8d,train,1200,2019-11-15,1")
    e = pc.encrypt("0")
    d = pc.decrypt(e)
    print('加密后的结果')
    print(e)
    print('解密后的结果')
    print(d)

    # 加密后的结果
    # e4f7ea4439ee83ea76d879adde3d6de5
    # 解密后的结果
    # 0

    #############TD02p1
    TD02p1 = pd.read_csv('/Users/ryanzheng/td/项目/联合外部建模项目/京东云联合建模/京东云项目的提数样本/label/TD02p1.csv')
    TD02p1_1 = TD02p1[['imei_md5', 'apply_time', 'target']]
    TD02p1_1["target"] = TD02p1_1["target"].map(str)
    TD02p1_1['merge'] = TD02p1_1["imei_md5"].map(str).str.cat([TD02p1_1["apply_time"], TD02p1_1["target"]], sep=',')
    TD02p1_1.to_csv('/Users/ryanzheng/td/项目/联合外部建模项目/京东云联合建模/京东云项目的提数样本/label/TD02p1_with_merge.csv')
    pc = prpcrypt('ecloud')
    TD02p1_1['merge'] = TD02p1_1['merge'].map(lambda x: pc.encrypt(x))
    TD02p1_1.rename(columns={'imei_md5': 'device_id'}, inplace=True)
    TD02p1_1[['device_id', 'apply_time', 'merge']].to_csv(
        '/Users/ryanzheng/td/项目/联合外部建模项目/京东云联合建模/京东云项目的提数样本/label/TD02p1_new.csv', index=False)
