import pymysql
import sqlalchemy
from sqlalchemy import create_engine

import pandas as pd

#보유하고 있는 파일(dataframe)을 DB에 저장
def pushToDB(file, password, dbName, tableName, ifexists):
    df = pd.read_csv(file)
    #mysql workbnech, vscode 연결
    conn = pymysql.connect(host='localhost', user='root', password=password, db=dbName, charset='utf8')
    cur = conn.cursor()
    #mysql, pymysql 연결 및 원하는 db 연동
    dbConnPath = f'mysql+pymysql://root:{password}@localhost/{dbName}'
    dbConn = create_engine(dbConnPath)
    conn = dbConn.connect()
    #dataframe을 해당 db table에 저장
    ## dataframe명.to_sql(name='table명', con=dbConn, )
    df.to_sql(name=tableName, con=dbConn, if_exists=ifexists, index=False)

###DB에서 dataframe으로 가져오기
def pullFromDB(password, dbName, tableName):
    conn = pymysql.connect(host='localhost', user='root', password=password, db=dbName, charset='utf8')
    df = pd.read_sql(f'SELECT * FROM {tableName}', con=conn)
    return df