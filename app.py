#!/usr/bin/python3
# -*- coding: utf-8 -*-
#ライブラリーインポート
import requests
from bs4 import BeautifulSoup
import sys
import MeCab
from time import sleep
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import sqlite3
from flask import Flask, redirect ,request,render_template,jsonify
from flask_bootstrap import Bootstrap
import json
from flask_ngrok import run_with_ngrok

#スクレイピングして文書加工
class Scr():
    def __init__(self, urls):
        self.urls=urls
#スクレイピング
    def geturl(self):
        all_text=[]
        for url in self.urls:
            r=requests.get(url)
            c=r.content
            soup=BeautifulSoup(c,"html.parser")
            article1_content=soup.find_all("p")
            temp=[]
            for con in article1_content:
                out=con.text
                temp.append(out)
            text=''.join(temp)
            all_text.append(text)
            sleep(1)
        return all_text

#メカブで形態素解析
def mplg(article):
    word_list = ""
    m=MeCab.Tagger()
    m1=m.parse(article)
    for row in m1.split("\n"):
        word =row.split("\t")[0]#タブ区切りになっている１つ目を取り出す。ここには形態素が格納されている
        if word == "EOS":
            break
        else:
            pos = row.split("\t")[1]#タブ区切りになっている2つ目を取り出す。ここには品詞が格納されている
            slice = pos[:2]
            if slice == "名詞":
                word_list = word_list +" "+ word
    return word_list

#文書類似度計算
class CalCos():
    def __init__(self,word_list):
        self.word=word_list
#tf-idf＆cos類似度で文書類似度算出
    def tfidf(self):
        docs = np.array(self.word)#Numpyの配列に変換する
        #単語を配列ベクトル化して、TF-IDFを計算する
        vecs = TfidfVectorizer(
                    token_pattern=u'(?u)\\b\\w+\\b'#文字列長が 1 の単語を処理対象に含めることを意味します。
                    ).fit_transform(docs)
        vecs = vecs.toarray()
        return vecs

    def cossim(self,v1,v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

#データベースアクセス
class Db():
    def __init__(self,dbname):
        self.db=dbname

    def db_input(self,article):
        #値をデータベースに格納
        conn=sqlite3.connect(self.db)
        c = conn.cursor()
        # executeメソッドでSQL文を実行する
        create_table = '''DROP TABLE IF EXISTS article_match;
                        create table article_match (text_1 verchar,text_2 verchar,match_rate double)'''
        c.executescript(create_table)
        # SQL文に値をセットする場合は，Pythonのformatメソッドなどは使わずに，
        # セットしたい場所に?を記述し，executeメソッドの第2引数に?に当てはめる値を
        # タプルで渡す．
        sql = 'insert into article_match (text_1, text_2, match_rate) values (?,?,?)'
        c.execute(sql, article)
        conn.commit()
        c.close()
        conn.close()

    def db_output(self):
        #データベースから値を抽出
        conn=sqlite3.connect(self.db)
        c = conn.cursor()
        select_sql = 'select * from article_match'
        c.execute(select_sql)
        match_rate=c.fetchone()
        conn.commit()
        c.close()
        conn.close()
        return match_rate

#こっから実装
nc = Flask(__name__)
bootstrap = Bootstrap(nc)
run_with_ngrok(nc)  # Start ngrok when app is run

@nc.route("/")
def home():
    return render_template('sr_all.html')

@nc.route('/output', methods=['POST'])
def output():
    #json形式でURLを受け取る
    url1 = request.json['url1']
    url2 = request.json['url2']

    word_list=[]
    url=[url1,url2]
    sc=Scr(url)
    texts=sc.geturl()
    for text in texts:
        word_list.append(mplg(text))

    wl=CalCos(word_list)
    vecs=wl.tfidf()
    match_rate=wl.cossim(vecs[1],vecs[0])
    #DB格納
    article = (url[0], url[1], match_rate)
    dbname = 'article.db'
    db=Db(dbname)
    db.db_input(article)
    match_rate=db.db_output()

    return_data = {"result":round(match_rate[2]*100,1)}
    return jsonify(ResultSet=json.dumps(return_data))

@nc.route('/output2', methods=['POST'])
def output2():
    #json形式でURLを受け取る
    text1 = request.json['text1']
    text2 = request.json['text2']
    word_list=[]
    texts=[text1,text2]

    for text in texts:
        word_list.append(mplg(text))

    wl=CalCos(word_list)
    vecs=wl.tfidf()
    match_rate=wl.cossim(vecs[1],vecs[0])
    #DB格納
    article = (texts[0], texts[1], match_rate)
    dbname = 'article.db'
    db=Db(dbname)
    db.db_input(article)
    match_rate=db.db_output()

    return_data = {"result":round(match_rate[2]*100,1)}
    return jsonify(ResultSet=json.dumps(return_data))

if __name__ == '__main__':
    nc.run()
