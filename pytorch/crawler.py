import requests
from bs4 import BeautifulSoup
import re
import pymssql
import time
import pandas as pd
import datetime
import random

base_url = 'http://www.310win.com/'
zcw_url = 'http://live.zgzcw.com/qb/'
Headers = {
    'User-Agent': "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)"}
"""
ZCW_Headers = {
    'User-Agent': "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.98 Safari/537.36 LBBROWSER",
    'Accept-Encoding': "gzip, deflate, sdch",
    'Accept-Language': "zh-CN,zh;q=0.8"
}
"""
CUP = '杯'
JB = '锦标'
ZZ = '足总'
black_list = ['球会友谊', '英社盾', '英锦赛']
league_5list = ['意甲', '法甲', '英超', '德甲', '西甲']
END = '完场'

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:61.0) Gecko/20100101 Firefox/61.0",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
    "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
    "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
    "Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
    "Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5"]

headers = {'User-Agent': random.choice(USER_AGENTS),
           "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
           "Accept-Encoding": "gzip, deflate, sdch",
           "Accept-Language": "zh-CN,zh;q=0.8,en;q=0.6",
           "Cache-Control": "no-cache"
           }


def test_http(ip_host):
    # 测试http代理是否有效

    try:
        url = 'https://www.baidu.com'
        proxies = {'http': ip_host}
        html = requests.get(url, headers=headers, proxies=proxies, timeout=5).text

        if html:
            print('HTTP @@代理有效(>_<): ', ip_host)
            return True
    except:
        print('HTTP @@代理无效(X_X!)：', ip_host)
        return False


def get_ip_list():
    print("正在获取代理列表...")
    #url = 'http://www.66ip.cn/index.html'
    #url = 'https://www.kuaidaili.com/free/'
    url = 'http://www.66ip.cn/'  #1.html
    headers = {
        "User-agent": random.choice(USER_AGENTS)
    }
    ip_list = []
    for tmp in range(1, 50):
        tmpurl = url+str(tmp)+'.html'
        #print(tmpurl)
        html = requests.get(tmpurl,headers=headers).text
        soup = BeautifulSoup(html, 'lxml')
        ips = soup.find(id='main').find_all('tr')
        for i in range(1, len(ips)):
            ip_info = ips[i]
            tds = ip_info.find_all('td')
            ip_host = tds[0].text + ':' + tds[1].text
            if test_http(ip_host):
                ip_list.append(tds[0].text + ':' + tds[1].text)
        print("代理列表抓取成功.", len(ip_list))
    return ip_list


def get_random_ip(ip_list=None):
    print("正在设置随机代理...")
    if not ip_list:
        ip_list = get_ip_list()
    proxy_list = []
    for ip in ip_list:
        proxy_list.append('https://' + ip)
    proxy_ip = random.choice(proxy_list)
    proxies = {'https': proxy_ip}
    print("代理设置成功.")
    return proxies


class MatchResult:
    home_all_total_match = 0
    home_all_total_win = 0
    home_all_total_draw = 0
    home_all_total_loss = 0
    home_all_total_get_goal = 0
    home_all_total_loss_goal = 0
    home_all_total_score = 0
    home_all_total_get_goal_rate = 0.0
    home_all_total_loss_goal_rate = 0.0

    home_total_match = 0
    home_total_win = 0
    home_total_draw = 0
    home_total_loss = 0
    home_total_get_goal = 0
    home_total_loss_goal = 0
    home_total_score = 0
    home_total_get_goal_rate = 0.0
    home_total_loss_goal_rate = 0.0

    home_six_match = 0
    home_six_win = 0
    home_six_draw = 0
    home_six_loss = 0
    home_six_get_goal = 0
    home_six_loss_goal = 0
    home_six_score = 0
    home_six_get_goal_rate = 0.0
    home_six_loss_goal_rate = 0.0

    guest_all_total_match = 0
    guest_all_total_win = 0
    guest_all_total_draw = 0
    guest_all_total_loss = 0
    guest_all_total_get_goal = 0
    guest_all_total_loss_goal = 0
    guest_all_total_score = 0
    guest_all_total_get_goal_rate = 0.0
    guest_all_total_loss_goal_rate = 0.0

    guest_total_match = 0
    guest_total_win = 0
    guest_total_draw = 0
    guest_total_loss = 0
    guest_total_get_goal = 0
    guest_total_loss_goal = 0
    guest_total_score = 0
    guest_total_get_goal_rate = 0.0
    guest_total_loss_goal_rate = 0.0

    guest_six_match = 0
    guest_six_win = 0
    guest_six_draw = 0
    guest_six_loss = 0
    guest_six_get_goal = 0
    guest_six_loss_goal = 0
    guest_six_score = 0
    guest_six_get_goal_rate = 0.0
    guest_six_loss_goal_rate = 0.0

    seq = ''
    league_title = ''
    home_team = ''
    guest_team = ''
    score = ''
    result = 0
    match_day = ''

    win = '1'
    draw = '1'
    loss = '1'


def has_attr_matchid(tag):
    return tag.has_attr('matchid')


def has_attr_cansale(tag):
    return tag.has_attr('cansale')


def has_attr_detail(bgcolor):
    return bgcolor and re.compile("#FFECEC").search(bgcolor)


def getresult(score):
    tmpscore = score.split('-')
    tmp = int(tmpscore[0]) - int(tmpscore[1])
    if tmp == 0:
        return 1
    elif tmp > 0:
        return 2

    return 0


def conn():
    connect = pymssql.connect(server='192.168.90.76', user='tdms_user', password='tdms_user@123!',
                              database='TDMS_SY')  # 服务器名,账户,密码,数据库名
    if connect:
        print("连接成功!")
    return connect


def getdetail_zcw(url):
    strhtml = requests.get(url, proxies=get_random_ip(ip_list), headers=headers)
    strhtml.encoding = 'utf-8'
    soup = BeautifulSoup(strhtml.text, 'lxml')
    div = soup.find_all("div", class_='only-team')
    if len(div) == 0:
        return None

    all_home_div = div[0].contents[3].contents[1]
    all_total_match = all_home_div.contents[3].contents[3].contents[1].text

    matchResult = MatchResult()
    # if home_trs[0].contents[3].text == '' or int(home_trs[0].contents[3].text) < 6:
    if all_total_match == '\xa0' or int(all_total_match) < 6:
        # print('home_trs', 'yes')
        return None  # home less six match will be ignore or no data

    matchResult.home_all_total_match = all_home_div.contents[3].contents[3].contents[1].text
    matchResult.home_all_total_win = all_home_div.contents[3].contents[3].contents[3].text
    matchResult.home_all_total_draw = all_home_div.contents[3].contents[3].contents[5].text
    matchResult.home_all_total_loss = all_home_div.contents[3].contents[3].contents[7].text
    matchResult.home_all_total_get_goal = all_home_div.contents[3].contents[3].contents[9].text
    matchResult.home_all_total_loss_goal = all_home_div.contents[3].contents[3].contents[11].text
    matchResult.home_all_total_score = all_home_div.contents[3].contents[3].contents[15].text

    home_div = div[0].contents[3].contents[3]
    home_total_match = home_div.contents[3].contents[3].contents[1].text
    if home_total_match == '0':
        return None  # data error
    matchResult.home_total_match = home_div.contents[3].contents[3].contents[1].text
    matchResult.home_total_win = home_div.contents[3].contents[3].contents[3].text
    matchResult.home_total_draw = home_div.contents[3].contents[3].contents[5].text
    matchResult.home_total_loss = home_div.contents[3].contents[3].contents[7].text
    matchResult.home_total_get_goal = home_div.contents[3].contents[3].contents[9].text
    matchResult.home_total_loss_goal = home_div.contents[3].contents[3].contents[11].text
    matchResult.home_total_score = home_div.contents[3].contents[3].contents[15].text

    # matchResult.home_six_match = home_div[3].contents[3].text
    # matchResult.home_six_win = home_div[3].contents[5].text
    # matchResult.home_six_draw = home_div[3].contents[7].text
    # matchResult.home_six_loss = home_div[3].contents[9].text
    # matchResult.home_six_get_goal = home_div[3].contents[11].text
    # matchResult.home_six_loss_goal = home_div[3].contents[13].text
    # matchResult.home_six_score = home_div[3].contents[15].text

    all_guest_div = div[0].contents[5].contents[1]
    guest_all_total_match = all_guest_div.contents[3].contents[3].contents[1].text
    if guest_all_total_match == '\xa0' or int(guest_all_total_match) < 6:
        # print('home_trs', 'yes')
        return None  # guest less six match will be ignore or no data
    matchResult.guest_all_total_match = all_guest_div.contents[3].contents[3].contents[1].text
    matchResult.guest_all_total_win = all_guest_div.contents[3].contents[3].contents[3].text
    matchResult.guest_all_total_draw = all_guest_div.contents[3].contents[3].contents[5].text
    matchResult.guest_all_total_loss = all_guest_div.contents[3].contents[3].contents[7].text
    matchResult.guest_all_total_get_goal = all_guest_div.contents[3].contents[3].contents[9].text
    matchResult.guest_all_total_loss_goal = all_guest_div.contents[3].contents[3].contents[11].text
    matchResult.guest_all_total_score = all_guest_div.contents[3].contents[3].contents[15].text

    guest_div = div[0].contents[5].contents[3]
    guest_total_match = guest_div.contents[3].contents[3].contents[1].text
    if guest_total_match == '0':
        return None  # data error

    matchResult.guest_total_match = guest_div.contents[3].contents[3].contents[1].text
    matchResult.guest_total_win = guest_div.contents[3].contents[3].contents[3].text
    matchResult.guest_total_draw = guest_div.contents[3].contents[3].contents[5].text
    matchResult.guest_total_loss = guest_div.contents[3].contents[3].contents[7].text
    matchResult.guest_total_get_goal = guest_div.contents[3].contents[3].contents[9].text
    matchResult.guest_total_loss_goal = guest_div.contents[3].contents[3].contents[11].text
    matchResult.guest_total_score = guest_div.contents[3].contents[3].contents[15].text

    # matchResult.guest_six_match = guest_trs[3].contents[3].text
    # matchResult.guest_six_win = guest_trs[3].contents[5].text
    # matchResult.guest_six_draw = guest_trs[3].contents[7].text
    # matchResult.guest_six_loss = guest_trs[3].contents[9].text
    # matchResult.guest_six_get_goal = guest_trs[3].contents[11].text
    # matchResult.guest_six_loss_goal = guest_trs[3].contents[13].text
    # matchResult.guest_six_score = guest_trs[3].contents[15].text
    # matchResult.guest_six_get_goal_rate = round(float(matchResult.guest_six_get_goal) / float(matchResult.guest_six_match),2)
    # matchResult.guest_six_loss_goal_rate = round(float(matchResult.guest_six_loss_goal) / float(matchResult.guest_six_match), 2)

    return matchResult


def getdetail(url):
    strhtml = requests.get(url, headers=Headers)
    strhtml.encoding = 'utf-8'
    soup = BeautifulSoup(strhtml.text, 'lxml')
    home_trs = soup.find_all("tr", align='middle', bgcolor='#FFECEC')
    if len(home_trs) == 0:
        return None

    matchResult = MatchResult()
    # if home_trs[0].contents[3].text == '' or int(home_trs[0].contents[3].text) < 6:
    if home_trs[0].contents[3].text == '\xa0' or int(home_trs[0].contents[3].text) < 6:
        # print('home_trs', 'yes')
        return None  # home less six match will be ignore or no data

    matchResult.home_all_total_match = home_trs[0].contents[3].text
    matchResult.home_all_total_win = home_trs[0].contents[5].text
    matchResult.home_all_total_draw = home_trs[0].contents[7].text
    matchResult.home_all_total_loss = home_trs[0].contents[9].text
    matchResult.home_all_total_get_goal = home_trs[0].contents[11].text
    matchResult.home_all_total_loss_goal = home_trs[0].contents[13].text
    matchResult.home_all_total_score = home_trs[0].contents[15].text
    # matchResult.home_all_total_get_goal_rate = round(float(matchResult.home_all_total_get_goal)/float(matchResult.home_all_total_match), 2)
    # matchResult.home_all_total_loss_goal_rate = round(float(matchResult.home_all_total_loss_goal)/float(matchResult.home_all_total_match), 2)

    if home_trs[1].contents[3].text == '0':
        return None  # data error
    matchResult.home_total_match = home_trs[1].contents[3].text
    matchResult.home_total_win = home_trs[1].contents[5].text
    matchResult.home_total_draw = home_trs[1].contents[7].text
    matchResult.home_total_loss = home_trs[1].contents[9].text
    matchResult.home_total_get_goal = home_trs[1].contents[11].text
    matchResult.home_total_loss_goal = home_trs[1].contents[13].text
    matchResult.home_total_score = home_trs[1].contents[15].text
    # matchResult.home_total_get_goal_rate = round(float(matchResult.home_total_get_goal)/float(matchResult.home_total_match), 2)
    # matchResult.home_total_loss_goal_rate = round(float(matchResult.home_total_loss_goal)/float(matchResult.home_total_match), 2)

    matchResult.home_six_match = home_trs[3].contents[3].text
    matchResult.home_six_win = home_trs[3].contents[5].text
    matchResult.home_six_draw = home_trs[3].contents[7].text
    matchResult.home_six_loss = home_trs[3].contents[9].text
    matchResult.home_six_get_goal = home_trs[3].contents[11].text
    matchResult.home_six_loss_goal = home_trs[3].contents[13].text
    matchResult.home_six_score = home_trs[3].contents[15].text
    # matchResult.home_six_get_goal_rate = round(float(matchResult.home_six_get_goal)/float(matchResult.home_six_match), 2)
    # matchResult.home_six_loss_goal_rate = round(float(matchResult.home_six_loss_goal)/float(matchResult.home_six_match),2)

    guest_trs = soup.find_all("tr", align='middle', bgcolor='#CCCCFF')
    if guest_trs[0].contents[3].text == '\xa0' or int(guest_trs[0].contents[3].text) < 6:
        # print('home_trs', 'yes')
        return None  # guest less six match will be ignore or no data
    matchResult.guest_all_total_match = guest_trs[0].contents[3].text
    matchResult.guest_all_total_win = guest_trs[0].contents[5].text
    matchResult.guest_all_total_draw = guest_trs[0].contents[7].text
    matchResult.guest_all_total_loss = guest_trs[0].contents[9].text
    matchResult.guest_all_total_get_goal = guest_trs[0].contents[11].text
    matchResult.guest_all_total_loss_goal = guest_trs[0].contents[13].text
    matchResult.guest_all_total_score = guest_trs[0].contents[15].text
    # matchResult.guest_all_total_get_goal_rate = round(float(matchResult.guest_all_total_get_goal) / float(matchResult.guest_all_total_match), 2)
    # matchResult.guest_all_total_loss_goal_rate = round(float(matchResult.guest_all_total_loss_goal) / float(matchResult.guest_all_total_match), 2)

    matchResult.guest_total_match = guest_trs[2].contents[3].text
    matchResult.guest_total_win = guest_trs[2].contents[5].text
    matchResult.guest_total_draw = guest_trs[2].contents[7].text
    matchResult.guest_total_loss = guest_trs[2].contents[9].text
    matchResult.guest_total_get_goal = guest_trs[2].contents[11].text
    matchResult.guest_total_loss_goal = guest_trs[2].contents[13].text
    matchResult.guest_total_score = guest_trs[2].contents[15].text
    # matchResult.guest_total_get_goal_rate = round(float(matchResult.guest_total_get_goal) / float(matchResult.guest_total_match), 2)
    # matchResult.guest_total_loss_goal_rate = round(float(matchResult.guest_total_loss_goal) / float(matchResult.guest_total_match), 2)

    matchResult.guest_six_match = guest_trs[3].contents[3].text
    matchResult.guest_six_win = guest_trs[3].contents[5].text
    matchResult.guest_six_draw = guest_trs[3].contents[7].text
    matchResult.guest_six_loss = guest_trs[3].contents[9].text
    matchResult.guest_six_get_goal = guest_trs[3].contents[11].text
    matchResult.guest_six_loss_goal = guest_trs[3].contents[13].text
    matchResult.guest_six_score = guest_trs[3].contents[15].text
    # matchResult.guest_six_get_goal_rate = round(float(matchResult.guest_six_get_goal) / float(matchResult.guest_six_match),2)
    # matchResult.guest_six_loss_goal_rate = round(float(matchResult.guest_six_loss_goal) / float(matchResult.guest_six_match), 2)

    return matchResult


def saveto_db_jcw(matchResultList):
    sql = 'insert into ball_all(match_day,seq,league_title,home_team,guest_team,score,result, \
    win,draw,loss, \
    home_all_total_match,home_all_total_win,home_all_total_draw,home_all_total_loss,home_all_total_get_goal,home_all_total_loss_goal,home_all_total_score,\
    home_total_match,home_total_win,home_total_draw,home_total_loss,home_total_get_goal,home_total_loss_goal,home_total_score,\
    home_six_match,home_six_win,home_six_draw,home_six_loss,home_six_get_goal,home_six_loss_goal,home_six_score,\
    guest_all_total_match,guest_all_total_win,guest_all_total_draw,guest_all_total_loss,guest_all_total_get_goal,guest_all_total_loss_goal, guest_all_total_score, \
    guest_total_match,guest_total_win,guest_total_draw,guest_total_loss,guest_total_get_goal,guest_total_loss_goal,guest_total_score, \
    guest_six_match,guest_six_win,guest_six_draw,guest_six_loss,guest_six_get_goal,guest_six_loss_goal,guest_six_score ) \
    values(%s, %s, %s, %s, %s, %s, %s, \
           %s, %s, %s, \
          %d, %d, %d, %d, %d, %d, %d,  \
          %d, %d, %d, %d, %d, %d, %d,  \
          %d, %d, %d, %d, %d, %d, %d,  \
          %d, %d, %d, %d, %d, %d, %d,  \
          %d, %d, %d, %d, %d, %d, %d,  \
          %d, %d, %d, %d, %d, %d, %d \
          )'

    data = []
    for matchResult in matchResultList:
        data.append((matchResult.match_day, matchResult.seq, matchResult.league_title, matchResult.home_team,
                     matchResult.guest_team, matchResult.score, matchResult.result,
                     matchResult.win, matchResult.draw, matchResult.loss,
                     matchResult.home_all_total_match, matchResult.home_all_total_win, matchResult.home_all_total_draw,
                     matchResult.home_all_total_loss, matchResult.home_all_total_get_goal,
                     matchResult.home_all_total_loss_goal, matchResult.home_all_total_score,

                     matchResult.home_total_match, matchResult.home_total_win, matchResult.home_total_draw,
                     matchResult.home_total_loss,
                     matchResult.home_total_get_goal, matchResult.home_total_loss_goal, matchResult.home_total_score,

                     matchResult.home_six_match, matchResult.home_six_win, matchResult.home_six_draw,
                     matchResult.home_six_loss,
                     matchResult.home_six_get_goal, matchResult.home_six_loss_goal, matchResult.home_six_score,

                     matchResult.guest_all_total_match, matchResult.guest_all_total_win,
                     matchResult.guest_all_total_draw, matchResult.guest_all_total_loss,
                     matchResult.guest_all_total_get_goal, matchResult.guest_all_total_loss_goal,
                     matchResult.guest_all_total_score,

                     matchResult.guest_total_match, matchResult.guest_total_win, matchResult.guest_total_draw,
                     matchResult.guest_total_loss,
                     matchResult.guest_total_get_goal, matchResult.guest_total_loss_goal, matchResult.guest_total_score,

                     matchResult.guest_six_match, matchResult.guest_six_win, matchResult.guest_six_draw,
                     matchResult.guest_six_loss,
                     matchResult.guest_six_get_goal, matchResult.guest_six_loss_goal, matchResult.guest_six_score
                     ))
    # print(data)
    con = conn()
    cursor = con.cursor()
    cursor.executemany(sql, data)
    con.commit()
    con.close()


def savetoDB(matchResultList):
    sql = 'insert into nball(match_day,seq,league_title,home_team,guest_team,score,result, \
    home_all_total_match,home_all_total_win,home_all_total_draw,home_all_total_loss,home_all_total_get_goal,home_all_total_loss_goal,home_all_total_score,\
    home_total_match,home_total_win,home_total_draw,home_total_loss,home_total_get_goal,home_total_loss_goal,home_total_score,\
    home_six_match,home_six_win,home_six_draw,home_six_loss,home_six_get_goal,home_six_loss_goal,home_six_score,\
    guest_all_total_match,guest_all_total_win,guest_all_total_draw,guest_all_total_loss,guest_all_total_get_goal,guest_all_total_loss_goal, guest_all_total_score, \
    guest_total_match,guest_total_win,guest_total_draw,guest_total_loss,guest_total_get_goal,guest_total_loss_goal,guest_total_score, \
    guest_six_match,guest_six_win,guest_six_draw,guest_six_loss,guest_six_get_goal,guest_six_loss_goal,guest_six_score ) \
    values(%s, %s, %s, %s, %s, %s, %s, \
          %d, %d, %d, %d, %d, %d, %d,  \
          %d, %d, %d, %d, %d, %d, %d,  \
          %d, %d, %d, %d, %d, %d, %d,  \
          %d, %d, %d, %d, %d, %d, %d,  \
          %d, %d, %d, %d, %d, %d, %d,  \
          %d, %d, %d, %d, %d, %d, %d \
          )'

    data = []
    for matchResult in matchResultList:
        data.append((matchResult.match_day, matchResult.seq, matchResult.league_title, matchResult.home_team,
                     matchResult.guest_team, matchResult.score, matchResult.result,

                     matchResult.home_all_total_match, matchResult.home_all_total_win, matchResult.home_all_total_draw,
                     matchResult.home_all_total_loss, matchResult.home_all_total_get_goal,
                     matchResult.home_all_total_loss_goal, matchResult.home_all_total_score,

                     matchResult.home_total_match, matchResult.home_total_win, matchResult.home_total_draw,
                     matchResult.home_total_loss,
                     matchResult.home_total_get_goal, matchResult.home_total_loss_goal, matchResult.home_total_score,

                     matchResult.home_six_match, matchResult.home_six_win, matchResult.home_six_draw,
                     matchResult.home_six_loss,
                     matchResult.home_six_get_goal, matchResult.home_six_loss_goal, matchResult.home_six_score,

                     matchResult.guest_all_total_match, matchResult.guest_all_total_win,
                     matchResult.guest_all_total_draw, matchResult.guest_all_total_loss,
                     matchResult.guest_all_total_get_goal, matchResult.guest_all_total_loss_goal,
                     matchResult.guest_all_total_score,

                     matchResult.guest_total_match, matchResult.guest_total_win, matchResult.guest_total_draw,
                     matchResult.guest_total_loss,
                     matchResult.guest_total_get_goal, matchResult.guest_total_loss_goal, matchResult.guest_total_score,

                     matchResult.guest_six_match, matchResult.guest_six_win, matchResult.guest_six_draw,
                     matchResult.guest_six_loss,
                     matchResult.guest_six_get_goal, matchResult.guest_six_loss_goal, matchResult.guest_six_score
                     ))
    # print(data)
    con = conn()
    cursor = con.cursor()
    cursor.executemany(sql, data)
    con.commit()
    con.close()


def get_zcw_result(fyear, fmonth, fday, tyear, tmonth, tday):
    begin = datetime.date(fyear, fmonth, fday)
    end = datetime.date(tyear, tmonth, tday)
    for i in range((end - begin).days + 1):
        day = begin + datetime.timedelta(days=i)
        day = str(day)
        url = 'http://live.zgzcw.com/qb/?date=' + day

        strhtml = requests.get(url, headers=headers)
        strhtml.encoding = 'utf-8'
        soup = BeautifulSoup(strhtml.text, 'lxml')

        matchResultList = []
        for tr in soup.find_all(has_attr_matchid):
            match_day = tr.contents[7]['date'][0:11]
            if match_day.strip() != day.strip():
                break  # 只取指定日期的

            seq = tr.contents[5].text
            league_title = tr.contents[3].text
            # print(seq, league_title)
            if CUP in league_title or JB in league_title or league_title in black_list or ZZ in league_title:
                continue  # filter cup match and 锦标赛 足总杯

            status = tr.contents[9].text
            if status not in END:
                continue  # filter cancel or delay match

            score = tr.contents[13].text
            result = getresult(score)

            win = 1.0
            draw = 1.0
            loss = 1.0
            score_tr = tr.contents[19].contents[1]
            # .contents[1].text
            if len(score_tr.contents) > 2:  # <1 是没有数据
                win = score_tr.contents[1].text
                draw = score_tr.contents[3].text
                loss = score_tr.contents[5].text

            home_team = tr.contents[11].contents[1].contents[7].text.replace('\n', '')
            guest_team = tr.contents[15].contents[1].contents[1].text.replace('\n', '')
            analysis_url = tr.contents[23].contents[4]['href']

            matchResult = getdetail_zcw(analysis_url)
            if matchResult is None:
                continue  # may be less six match or zero match
            # time.sleep(0.5)

            matchResult.seq = seq
            matchResult.league_title = league_title
            matchResult.home_team = home_team
            matchResult.guest_team = guest_team
            matchResult.score = score
            matchResult.result = result
            matchResult.match_day = day.replace('-', '')

            matchResult.win = win
            matchResult.draw = draw
            matchResult.loss = loss

            matchResultList.append(matchResult)

        saveto_db_jcw(matchResultList)
        print(day, 'jcw_end')
        print(datetime.datetime.now())
        time.sleep(63)


def get_result(fyear, fmonth, fday, tyear, tmonth, tday):
    begin = datetime.date(fyear, fmonth, fday)
    end = datetime.date(tyear, tmonth, tday)
    for i in range((end - begin).days + 1):
        day = begin + datetime.timedelta(days=i)
        day = str(day)
        url = base_url + 'buy/jingcai.aspx?typeID=105&oddstype=2&date=' + day
        strhtml = requests.get(url, headers=Headers)
        strhtml.encoding = 'utf-8'
        soup = BeautifulSoup(strhtml.text, 'lxml')
        matchResultList = []
        for tr in soup.find_all(has_attr_cansale):
            seq = tr.contents[1].text
            league_title = tr.contents[3].text
            # print(seq, league_title)
            if CUP in league_title or JB in league_title or league_title in black_list:
                continue  # filter cup match and 锦标赛

            status = tr.contents[7].text
            if status != END:
                continue  # filter cancel or delay match

            score = tr.contents[13].text
            result = getresult(score)
            home_team = tr.contents[9].text.replace('\n', '')
            guest_team = tr.contents[15].text.replace('\n', '')
            analysis_url = base_url + tr.contents[21].contents[2]['href']

            matchResult = getdetail(analysis_url)
            if matchResult is None:
                continue  # may be less six match or zero match
            # time.sleep(0.5)

            matchResult.seq = seq
            matchResult.league_title = league_title
            matchResult.home_team = home_team
            matchResult.guest_team = guest_team
            matchResult.score = score
            matchResult.result = result
            matchResult.match_day = day.replace('-', '')

            matchResultList.append(matchResult)

        savetoDB(matchResultList)
        print(day, 'end')
        time.sleep(3)


def get_match_to_predict(year, month, day):
    begin = datetime.date(year, month, day)
    end = datetime.date(year, month, day)
    matchResultList = []
    for i in range((end - begin).days + 1):
        day = begin + datetime.timedelta(days=i)
        day = str(day)
        url = base_url + 'buy/jingcai.aspx?typeID=105&oddstype=2&date=' + day
        strhtml = requests.get(url, headers=Headers)
        strhtml.encoding = 'utf-8'
        soup = BeautifulSoup(strhtml.text, 'lxml')

        for tr in soup.find_all(has_attr_cansale):
            seq = tr.contents[1].text
            league_title = tr.contents[3].text
            # print(seq, league_title)
            if CUP in league_title or JB in league_title or league_title in black_list:
                continue  # filter cup match and 锦标赛

            home_team = tr.contents[9].text.replace('\n', '')
            guest_team = tr.contents[15].text.replace('\n', '')
            analysis_url = base_url + tr.contents[21].contents[2]['href']

            matchResult = getdetail(analysis_url)
            if matchResult is None:
                continue  # may be less six match or zero match

            # match_day = day.replace('-', '')
            match_day = str(tr.contents[7])[16:26].replace('-', '')

            # home_all_rate = round(float(matchResult.home_all_total_score) / float(matchResult.home_all_total_match), 2)
            # home_all_total_get_goal_rate = round(float(matchResult.home_all_total_get_goal) / float(matchResult.home_all_total_match), 2)
            # home_all_total_loss_goal_rate = round(float(matchResult.home_all_total_loss_goal) / float(matchResult.home_all_total_match), 2)

            home_rate = round(float(matchResult.home_total_score) / float(matchResult.home_total_match), 2)
            home_total_get_goal_rate = round(
                float(matchResult.home_total_get_goal) / float(matchResult.home_total_match), 2)
            home_total_loss_goal_rate = round(
                float(matchResult.home_total_loss_goal) / float(matchResult.home_total_match), 2)

            home_six_rate = round(float(matchResult.home_six_score) / float(matchResult.home_six_match), 2)
            home_six_get_goal_rate = round(float(matchResult.home_six_get_goal) / float(matchResult.home_six_match), 2)
            home_six_loss_goal_rate = round(float(matchResult.home_six_loss_goal) / float(matchResult.home_six_match),
                                            2)

            # guest_all_rate = round(float(matchResult.guest_all_total_score) / float(matchResult.guest_all_total_match),2)
            # guest_all_total_get_goal_rate = round(float(matchResult.guest_all_total_get_goal) / float(matchResult.guest_all_total_match),2)
            # guest_all_total_loss_goal_rate = round(float(matchResult.guest_all_total_loss_goal) / float(matchResult.guest_all_total_match),2)

            guest_rate = round(float(matchResult.guest_total_score) / float(matchResult.guest_total_match), 2)
            guest_total_get_goal_rate = round(
                float(matchResult.guest_total_get_goal) / float(matchResult.guest_total_match), 2)
            guest_total_loss_goal_rate = round(
                float(matchResult.guest_total_loss_goal) / float(matchResult.guest_total_match), 2)

            guest_six_rate = round(float(matchResult.guest_six_score) / float(matchResult.guest_six_match), 2)
            guest_six_get_goal_rate = round(float(matchResult.guest_six_get_goal) / float(matchResult.guest_six_match),
                                            2)
            guest_six_loss_goal_rate = round(
                float(matchResult.guest_six_loss_goal) / float(matchResult.guest_six_match), 2)

            match = [match_day, seq, league_title, home_team, guest_team,
                     # home_all_rate, home_all_total_get_goal_rate, home_all_total_loss_goal_rate,
                     home_rate, home_total_get_goal_rate, home_total_loss_goal_rate,
                     home_six_rate, home_six_get_goal_rate, home_six_loss_goal_rate,
                     # guest_all_rate, guest_all_total_get_goal_rate, guest_all_total_loss_goal_rate,
                     guest_rate, guest_total_get_goal_rate, guest_total_loss_goal_rate,
                     guest_six_rate, guest_six_get_goal_rate, guest_six_loss_goal_rate]

            matchResultList.append(match)

    # print(matchResultList)
    columns = ["match_day", "seq", "league_title", "home_team", "guest_team",
               # "home_all_rate", "home_all_total_get_goal_rate", "home_all_total_loss_goal_rate",
               "home_rate", "home_total_get_goal_rate", "home_total_loss_goal_rate",
               "home_six_rate", "home_six_get_goal_rate", "home_six_loss_goal_rate",
               # "guest_all_rate", "guest_all_total_get_goal_rate", "guest_all_total_loss_goal_rate",
               "guest_rate", "guest_total_get_goal_rate", "guest_total_loss_goal_rate",
               "guest_six_rate", "guest_six_get_goal_rate", "guest_six_loss_goal_rate"
               ]

    dt = pd.DataFrame(matchResultList, columns=columns)
    root = 'D:\\AI\\ball\\'
    filename = "predict" + day + ".csv"
    dt.to_csv(root + filename, encoding='utf_8_sig')

    print('end')


# 增加比分的获取
def get_match_to_val(fyear, fmonth, fday, tyear, tmonth, tday):
    begin = datetime.date(fyear, fmonth, fday)
    end = datetime.date(tyear, tmonth, tday)
    matchResultList = []
    for i in range((end - begin).days + 1):
        day = begin + datetime.timedelta(days=i)
        day = str(day)
        url = base_url + 'buy/jingcai.aspx?typeID=105&oddstype=2&date=' + day
        strhtml = requests.get(url, headers=Headers)
        strhtml.encoding = 'utf-8'
        soup = BeautifulSoup(strhtml.text, 'lxml')

        for tr in soup.find_all(has_attr_cansale):
            seq = tr.contents[1].text
            league_title = tr.contents[3].text
            # print(seq, league_title)
            if CUP in league_title or JB in league_title or league_title in black_list:
                continue  # filter cup match and 锦标赛

            status = tr.contents[7].text
            if status != END:
                continue  # filter cancel or delay match

            score = tr.contents[13].text
            result = getresult(score)

            home_team = tr.contents[9].text.replace('\n', '')
            guest_team = tr.contents[15].text.replace('\n', '')
            analysis_url = base_url + tr.contents[21].contents[2]['href']

            score_tr = tr.contents[23].contents[0].contents[0]
            # print(score_tr)
            # print(len(score_tr.contents))
            win = 1.0
            draw = 1.0
            loss = 1.0
            if len(score_tr.contents) > 2:  # <1 是未开售胜平负玩法
                win = score_tr.contents[1].contents[0].text
                draw = score_tr.contents[2].contents[0].text
                loss = score_tr.contents[3].contents[0].text

            matchResult = getdetail(analysis_url)
            if matchResult is None:
                continue  # may be less six match or zero match

            # match_day = day.replace('-', '')
            match_day = str(tr.contents[7])[16:26].replace('-', '')

            # home_all_rate = round(float(matchResult.home_all_total_score) / float(matchResult.home_all_total_match), 2)
            # home_all_total_get_goal_rate = round(float(matchResult.home_all_total_get_goal) / float(matchResult.home_all_total_match), 2)
            # home_all_total_loss_goal_rate = round(float(matchResult.home_all_total_loss_goal) / float(matchResult.home_all_total_match), 2)

            home_rate = round(float(matchResult.home_total_score) / float(matchResult.home_total_match), 2)
            home_total_get_goal_rate = round(
                float(matchResult.home_total_get_goal) / float(matchResult.home_total_match), 2)
            home_total_loss_goal_rate = round(
                float(matchResult.home_total_loss_goal) / float(matchResult.home_total_match), 2)

            home_six_rate = round(float(matchResult.home_six_score) / float(matchResult.home_six_match), 2)
            home_six_get_goal_rate = round(float(matchResult.home_six_get_goal) / float(matchResult.home_six_match), 2)
            home_six_loss_goal_rate = round(float(matchResult.home_six_loss_goal) / float(matchResult.home_six_match),
                                            2)

            # guest_all_rate = round(float(matchResult.guest_all_total_score) / float(matchResult.guest_all_total_match),2)
            # guest_all_total_get_goal_rate = round(float(matchResult.guest_all_total_get_goal) / float(matchResult.guest_all_total_match),2)
            # guest_all_total_loss_goal_rate = round(float(matchResult.guest_all_total_loss_goal) / float(matchResult.guest_all_total_match),2)

            guest_rate = round(float(matchResult.guest_total_score) / float(matchResult.guest_total_match), 2)
            guest_total_get_goal_rate = round(
                float(matchResult.guest_total_get_goal) / float(matchResult.guest_total_match), 2)
            guest_total_loss_goal_rate = round(
                float(matchResult.guest_total_loss_goal) / float(matchResult.guest_total_match), 2)

            guest_six_rate = round(float(matchResult.guest_six_score) / float(matchResult.guest_six_match), 2)
            guest_six_get_goal_rate = round(float(matchResult.guest_six_get_goal) / float(matchResult.guest_six_match),
                                            2)
            guest_six_loss_goal_rate = round(
                float(matchResult.guest_six_loss_goal) / float(matchResult.guest_six_match), 2)

            match = [match_day, seq, league_title, home_team, guest_team,
                     result, win, draw, loss,
                     # home_all_rate, home_all_total_get_goal_rate, home_all_total_loss_goal_rate,
                     home_rate, home_total_get_goal_rate, home_total_loss_goal_rate,
                     home_six_rate, home_six_get_goal_rate, home_six_loss_goal_rate,
                     # guest_all_rate, guest_all_total_get_goal_rate, guest_all_total_loss_goal_rate,
                     guest_rate, guest_total_get_goal_rate, guest_total_loss_goal_rate,
                     guest_six_rate, guest_six_get_goal_rate, guest_six_loss_goal_rate]

            matchResultList.append(match)

    # print(matchResultList)
    columns = ["match_day", "seq", "league_title", "home_team", "guest_team",
               "result", "win", "draw", "loss",
               # "home_all_rate", "home_all_total_get_goal_rate", "home_all_total_loss_goal_rate",
               "home_rate", "home_total_get_goal_rate", "home_total_loss_goal_rate",
               "home_six_rate", "home_six_get_goal_rate", "home_six_loss_goal_rate",
               # "guest_all_rate", "guest_all_total_get_goal_rate", "guest_all_total_loss_goal_rate",
               "guest_rate", "guest_total_get_goal_rate", "guest_total_loss_goal_rate",
               "guest_six_rate", "guest_six_get_goal_rate", "guest_six_loss_goal_rate"
               ]

    dt = pd.DataFrame(matchResultList, columns=columns)
    root = 'D:\\AI\\ball\\'
    # filename = "val" + day + ".csv"
    # dt.to_csv(root + filename, encoding='utf_8_sig')
    filename = "val" + day + ".xlsx"
    dt.to_excel(root + filename, encoding='utf_8_sig')

    print('get_match_to_val end')


# 没有五大联赛
def get_match_to_predict_no_5league(year, month, day):
    begin = datetime.date(year, month, day)
    end = datetime.date(year, month, day)
    matchResultList = []
    for i in range((end - begin).days + 1):
        day = begin + datetime.timedelta(days=i)
        day = str(day)
        url = base_url + 'buy/jingcai.aspx?typeID=105&oddstype=2&date=' + day
        strhtml = requests.get(url, headers=Headers)
        strhtml.encoding = 'utf-8'
        soup = BeautifulSoup(strhtml.text, 'lxml')

        for tr in soup.find_all(has_attr_cansale):
            seq = tr.contents[1].text
            league_title = tr.contents[3].text
            # print(seq, league_title)
            if CUP in league_title or JB in league_title or league_title in black_list or league_title in league_5list:
                continue  # filter cup match and 锦标赛

            home_team = tr.contents[9].text.replace('\n', '')
            guest_team = tr.contents[15].text.replace('\n', '')
            analysis_url = base_url + tr.contents[21].contents[2]['href']

            matchResult = getdetail(analysis_url)
            if matchResult is None:
                continue  # may be less six match or zero match

            # match_day = day.replace('-', '')
            match_day = str(tr.contents[7])[16:26].replace('-', '')

            # home_all_rate = round(float(matchResult.home_all_total_score) / float(matchResult.home_all_total_match), 2)
            # home_all_total_get_goal_rate = round(float(matchResult.home_all_total_get_goal) / float(matchResult.home_all_total_match), 2)
            # home_all_total_loss_goal_rate = round(float(matchResult.home_all_total_loss_goal) / float(matchResult.home_all_total_match), 2)

            home_rate = round(float(matchResult.home_total_score) / float(matchResult.home_total_match), 2)
            home_total_get_goal_rate = round(
                float(matchResult.home_total_get_goal) / float(matchResult.home_total_match), 2)
            home_total_loss_goal_rate = round(
                float(matchResult.home_total_loss_goal) / float(matchResult.home_total_match), 2)

            home_six_rate = round(float(matchResult.home_six_score) / float(matchResult.home_six_match), 2)
            home_six_get_goal_rate = round(float(matchResult.home_six_get_goal) / float(matchResult.home_six_match), 2)
            home_six_loss_goal_rate = round(float(matchResult.home_six_loss_goal) / float(matchResult.home_six_match),
                                            2)

            # guest_all_rate = round(float(matchResult.guest_all_total_score) / float(matchResult.guest_all_total_match),2)
            # guest_all_total_get_goal_rate = round(float(matchResult.guest_all_total_get_goal) / float(matchResult.guest_all_total_match),2)
            # guest_all_total_loss_goal_rate = round(float(matchResult.guest_all_total_loss_goal) / float(matchResult.guest_all_total_match),2)

            guest_rate = round(float(matchResult.guest_total_score) / float(matchResult.guest_total_match), 2)
            guest_total_get_goal_rate = round(
                float(matchResult.guest_total_get_goal) / float(matchResult.guest_total_match), 2)
            guest_total_loss_goal_rate = round(
                float(matchResult.guest_total_loss_goal) / float(matchResult.guest_total_match), 2)

            guest_six_rate = round(float(matchResult.guest_six_score) / float(matchResult.guest_six_match), 2)
            guest_six_get_goal_rate = round(float(matchResult.guest_six_get_goal) / float(matchResult.guest_six_match),
                                            2)
            guest_six_loss_goal_rate = round(
                float(matchResult.guest_six_loss_goal) / float(matchResult.guest_six_match), 2)

            match = [match_day, seq, league_title, home_team, guest_team,
                     # home_all_rate, home_all_total_get_goal_rate, home_all_total_loss_goal_rate,
                     home_rate, home_total_get_goal_rate, home_total_loss_goal_rate,
                     home_six_rate, home_six_get_goal_rate, home_six_loss_goal_rate,
                     # guest_all_rate, guest_all_total_get_goal_rate, guest_all_total_loss_goal_rate,
                     guest_rate, guest_total_get_goal_rate, guest_total_loss_goal_rate,
                     guest_six_rate, guest_six_get_goal_rate, guest_six_loss_goal_rate]

            matchResultList.append(match)

    # print(matchResultList)
    columns = ["match_day", "seq", "league_title", "home_team", "guest_team",
               # "home_all_rate", "home_all_total_get_goal_rate", "home_all_total_loss_goal_rate",
               "home_rate", "home_total_get_goal_rate", "home_total_loss_goal_rate",
               "home_six_rate", "home_six_get_goal_rate", "home_six_loss_goal_rate",
               # "guest_all_rate", "guest_all_total_get_goal_rate", "guest_all_total_loss_goal_rate",
               "guest_rate", "guest_total_get_goal_rate", "guest_total_loss_goal_rate",
               "guest_six_rate", "guest_six_get_goal_rate", "guest_six_loss_goal_rate"
               ]

    dt = pd.DataFrame(matchResultList, columns=columns)
    root = 'D:\\AI\\ball\\'
    filename = "predict_no5league_" + day + ".csv"
    dt.to_csv(root + filename, encoding='utf_8_sig')

    print('end')


def main():
    global ip_list
    ip_list = get_ip_list()
    # get_result(2015, 4, 26, 2015, 12, 31)
    # get_match_to_predict(2020, 6, 30)
    # get_match_to_val(2020, 6, 1, 2020, 7, 9)
    # get_match_to_predict_no_5league(2020, 6, 30)
    get_zcw_result(2019, 4, 8, 2019, 4, 8)  # 2019, 4, 8, 2019, 12, 31


if __name__ == '__main__':
    main()
