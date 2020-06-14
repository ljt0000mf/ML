import requests
from bs4 import BeautifulSoup

params = {
        'enc': 'ISO-8859-1'
    }

base_url = 'http://www.310win.com/'

"""
url = 'http://www.310win.com/buy/jingcai.aspx?typeID=105&oddstype=2&date=2020-6-4'
strhtml = requests.get(url)        #Get方式获取网页数据
#print(strhtml.text)
soup = BeautifulSoup(strhtml.text, 'lxml')
#data = soup.select('#main>div>div.mtop.firstMod.clearfix>div.centerBox>ul.newsList>li>a')
data = soup.select('#MatchTable')
print(data)
# print(soup)
"""
#url2 = 'http://www.310win.com/analysis/1745162.htm'
#strhtml2 = requests.get(url2)
#strhtml2.encoding='utf-8'
#print(strhtml2.encoding)

#soup2 = BeautifulSoup(strhtml2.text, 'lxml')
#soup2 = BeautifulSoup(strhtml2.text, 'lxml')
#print(soup2)

def main():
    url = base_url+'buy/jingcai.aspx?typeID=105&oddstype=2&date=2020-6-4'
    strhtml = requests.get(url)
    strhtml.encoding = 'utf-8'
    soup = BeautifulSoup(strhtml.text, 'lxml')
    data = soup.select('#MatchTable')
    data.select('')
    print(data)

if __name__ == '__main__':
    main()

