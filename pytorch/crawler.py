import requests
from bs4 import BeautifulSoup

url = 'http://www.310win.com/buy/jingcai.aspx?typeID=105&oddstype=2&date=2020-6-4'
strhtml = requests.get(url)        #Get方式获取网页数据
#print(strhtml.text)
soup = BeautifulSoup(strhtml.text, 'lxml')
#data = soup.select('#main>div>div.mtop.firstMod.clearfix>div.centerBox>ul.newsList>li>a')
data = soup.select('#MatchTable')
print(data)