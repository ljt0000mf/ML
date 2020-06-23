import requests
from bs4 import BeautifulSoup
import re
import pymssql
import datetime
import time
import pandas as pd

base_url = 'http://www.310win.com/'
Headers = {
    'User-Agent': "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)"}
CUP = '杯'
JB = '锦标'
END = '完场'


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

                     matchResult.home_total_match, matchResult.home_total_win,matchResult.home_total_draw,matchResult.home_total_loss,
                     matchResult.home_total_get_goal, matchResult.home_total_loss_goal, matchResult.home_total_score,

                     matchResult.home_six_match, matchResult.home_six_win, matchResult.home_six_draw,matchResult.home_six_loss,
                     matchResult.home_six_get_goal, matchResult.home_six_loss_goal, matchResult.home_six_score,

                     matchResult.guest_all_total_match,matchResult.guest_all_total_win,matchResult.guest_all_total_draw,matchResult.guest_all_total_loss,
                     matchResult.guest_all_total_get_goal, matchResult.guest_all_total_loss_goal,matchResult.guest_all_total_score,

                     matchResult.guest_total_match,matchResult.guest_total_win,matchResult.guest_total_draw,matchResult.guest_total_loss,
                     matchResult.guest_total_get_goal, matchResult.guest_total_loss_goal,matchResult.guest_total_score,

                     matchResult.guest_six_match, matchResult.guest_six_win, matchResult.guest_six_draw,matchResult.guest_six_loss,
                     matchResult.guest_six_get_goal, matchResult.guest_six_loss_goal, matchResult.guest_six_score
                     ))
    # print(data)
    con = conn()
    cursor = con.cursor()
    cursor.executemany(sql, data)
    con.commit()
    con.close()


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
            if CUP in league_title or JB in league_title:
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
            #time.sleep(0.5)

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
            if CUP in league_title or JB in league_title:
                continue  # filter cup match and 锦标赛

            home_team = tr.contents[9].text.replace('\n', '')
            guest_team = tr.contents[15].text.replace('\n', '')
            analysis_url = base_url + tr.contents[21].contents[2]['href']

            matchResult = getdetail(analysis_url)
            if matchResult is None:
                continue  # may be less six match or zero match

            # match_day = day.replace('-', '')
            match_day = str(tr.contents[7])[16:26].replace('-', '')

            home_all_rate = round(float(matchResult.home_all_total_score) / float(matchResult.home_all_total_match), 2)
            home_all_total_get_goal_rate = round(float(matchResult.home_all_total_get_goal) / float(matchResult.home_all_total_match), 2)
            home_all_total_loss_goal_rate = round(float(matchResult.home_all_total_loss_goal) / float(matchResult.home_all_total_match), 2)

            home_rate = round(float(matchResult.home_total_score) / float(matchResult.home_total_match), 2)
            home_total_get_goal_rate = round(float(matchResult.home_total_get_goal) / float(matchResult.home_total_match), 2)
            home_total_loss_goal_rate = round(float(matchResult.home_total_loss_goal) / float(matchResult.home_total_match), 2)

            home_six_rate = round(float(matchResult.home_six_score) / float(matchResult.home_six_match), 2)
            home_six_get_goal_rate = round(float(matchResult.home_six_get_goal) / float(matchResult.home_six_match), 2)
            home_six_loss_goal_rate = round(float(matchResult.home_six_loss_goal) / float(matchResult.home_six_match), 2)

            guest_all_rate = round(float(matchResult.guest_all_total_score) / float(matchResult.guest_all_total_match),2)
            guest_all_total_get_goal_rate = round(float(matchResult.guest_all_total_get_goal) / float(matchResult.guest_all_total_match),2)
            guest_all_total_loss_goal_rate = round(float(matchResult.guest_all_total_loss_goal) / float(matchResult.guest_all_total_match),2)

            guest_rate = round(float(matchResult.guest_total_score) / float(matchResult.guest_total_match), 2)
            guest_total_get_goal_rate = round(float(matchResult.guest_total_get_goal) / float(matchResult.guest_total_match), 2)
            guest_total_loss_goal_rate = round(float(matchResult.guest_total_loss_goal) / float(matchResult.guest_total_match), 2)

            guest_six_rate = round(float(matchResult.guest_six_score) / float(matchResult.guest_six_match), 2)
            guest_six_get_goal_rate = round(float(matchResult.guest_six_get_goal) / float(matchResult.guest_six_match), 2)
            guest_six_loss_goal_rate = round(float(matchResult.guest_six_loss_goal) / float(matchResult.guest_six_match),2)

            match = [match_day, seq, league_title, home_team, guest_team,
                     home_all_rate, home_all_total_get_goal_rate, home_all_total_loss_goal_rate,
                     home_rate, home_total_get_goal_rate, home_total_loss_goal_rate,
                     home_six_rate, home_six_get_goal_rate, home_six_loss_goal_rate,
                     guest_all_rate, guest_all_total_get_goal_rate, guest_all_total_loss_goal_rate,
                     guest_rate, guest_total_get_goal_rate, guest_total_loss_goal_rate,
                     guest_six_rate, guest_six_get_goal_rate, guest_six_loss_goal_rate]

            matchResultList.append(match)

    # print(matchResultList)
    columns = ["match_day", "seq", "league_title", "home_team", "guest_team",
               "home_all_rate", "home_all_total_get_goal_rate", "home_all_total_loss_goal_rate",
               "home_rate", "home_total_get_goal_rate", "home_total_loss_goal_rate",
               "home_six_rate", "home_six_get_goal_rate", "home_six_loss_goal_rate",
               "guest_all_rate", "guest_all_total_get_goal_rate", "guest_all_total_loss_goal_rate",
               "guest_rate", "guest_total_get_goal_rate", "guest_total_loss_goal_rate",
               "guest_six_rate", "guest_six_get_goal_rate", "guest_six_loss_goal_rate"
               ]

    dt = pd.DataFrame(matchResultList, columns=columns)
    root = 'D:\\AI\\ball\\'
    filename = "predict" + day + ".csv"
    dt.to_csv(root + filename, encoding='utf_8_sig')

    print('end')


def main():
    #get_result(2016, 9, 15, 2016, 12, 31)
    get_match_to_predict(2020, 6, 23)


if __name__ == '__main__':
    main()
