import matplotlib.pyplot as plt
import re
import requests
import datetime
import time
import pandas as pd
import json
from bs4 import BeautifulSoup
from lxml import etree
query_list = ['aapl', 'apple+inc']
filter_keywords = ['aapl', 'iphone', 'ipad', 'apple', 'app', 'stock', 'aal', 'mac', 'steve']

# Set the Start Date
start_month = 1
start_day = 1
start_year = 2014

# Set the End Date
end_month = 12
end_day = 31
end_year = 2018

# NYTimes API Key
API_Key = "YourAPI"
Interval = 10

# Get the date range between two dates
def daterange(start, end):
    for i in range((end - start).days + 1):
        yield (start + datetime.timedelta(i))

# Get the date range between two dates for given intervals
def daterangeintv(start, end, intv):
    for i in range(intv):
        yield (start + (end - start) / intv * i)
    yield end

start_date = datetime.date(year=start_year, month=start_month, day=start_day)
end_date = datetime.date(year=end_year, month=end_month, day=end_day)
Headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}
Stock_Data = pd.read_csv('https://www.kaggle.com/datasets/krupalpatel07/apple-stock-data/download')
Stock_Data['Date'] = pd.to_datetime(Stock_Data['Date'])
Stock_Data = Stock_Data[(Stock_Data['Date'] >= pd.Timestamp(start_date)) & 
                        (Stock_Data['Date'] <= pd.Timestamp(end_date))]
Stock_Data = Stock_Data[['Date', 'Close', 'Volume']]
Stock_Data.set_index('Date', inplace=True)
Stock_Data.to_csv("Stock_Data.csv")
PD_Headers = pd.DataFrame({'Date': [], 'Headline': []})

for Date in daterange(start_date, end_date):
    for query in query_list:
        URL = 'https://www.business-standard.com/advance-search?type=news&c-q=q&q='+query+'&c-range=range&range=bwn_dates&from_date='+Date.strftime("%d-%m-20%y")+'&to_date='+Date.strftime("%d-%m-20%y")
        raw_html = requests.get(URL, headers=Headers)
        soup = BeautifulSoup(raw_html.text, 'lxml')
        raw = soup.find("ul", class_="listing")
        headline_list = raw.find_all("a", href=re.compile("/article/"))

        dates = []
        headlines = []

        for elem in headline_list:
            headlines.append(elem.text)

        for elem in headline_list:
            dates.append(Date)

        News = pd.DataFrame({'Date': dates, 'Headline': headlines})
        PD_Headers = pd.concat([PD_Headers, News])

PD_Headers.drop_duplicates(['Headline'], keep='last')
PD_Headers = PD_Headers[PD_Headers['Headline'].str.contains('|'.join(filter_keywords), case=False)]
PD_Headers.to_csv('files/'+query_list[0]+'_News_BS.csv')

# New York Times News Scraping
Datelist = list(daterangeintv(start_date, end_date, Interval))
PD_Headers = pd.DataFrame({'Date': [], 'Headline': []})

def access_api(query, page, start_date, end_date):
    time.sleep(1)
    URL = 'http://api.nytimes.com/svc/search/v2/articlesearch.json?q='+query+'&sort=relevance&fq='+query+'&page='+str(page)+'&api-key='+API_Key+'&begin_date='+start_date.strftime("%Y%m%d")+'&end_date='+end_date.strftime("%Y%m%d")

    raw_html = None
    content = None
    i = 0

    while i<3 and (content is None or raw_html.status_code!=200):
        try:
            raw_html = requests.get(URL)
            data = json.loads(raw_html.content.decode("utf-8"))
            content = data["response"]
        except ValueError:
            raise ValueError
        except KeyError:
            continue
        i += 1

    return content

for i in range(1, Interval):
    for query in query_list:
        dates = []
        headlines = []
        page = 0

        Data = access_api(query, page, Datelist[i-1], Datelist[i])
        while Data["meta"]["hits"] > 1000:
            Data = access_api(query, page, Datelist[i-1], Datelist[i])
        while page * 10 < Data["meta"]["hits"] and (page + 1) < 100:
            Data = access_api(query, page, Datelist[i-1], Datelist[i])
            for doc in Data["docs"]:
                headlines.append(doc['headline']['main'])
                dates.append(doc['pub_date'][0:10])
            page += 1

        News = pd.DataFrame({'Date': dates, 'Headline': headlines})
        PD_Headers = pd.concat([PD_Headers, News])

PD_Headers.drop_duplicates(['Headline'], keep='last')
PD_Headers = PD_Headers[PD_Headers['Headline'].str.contains('|'.join(filter_keywords), case=False)]
PD_Headers.to_csv('files/'+query_list[0]+'_News_NYTimes.csv')
PD_Headers.to_csv("News_NYTimes.csv")

# Financial Times News Scraping
PD_Headers = pd.DataFrame({'Date': [], 'Headline': []})

for Date in daterange(start_date, end_date):
    for query in query_list:
        for page in range(1, 3):
            URL = 'https://www.ft.com/search?expandRefinements=true&q='+query+'&concept=a39a4558-f562-4dca-8774-000246e6eebe&dateFrom='+Date.strftime("20%y-%m-%d")+'&dateTo='+Date.strftime("20%y-%m-%d")+'&page='+str(page)
            raw_html = requests.get(URL, headers=Headers)
            soup = BeautifulSoup(raw_html.text, 'html.parser')
            headline_list = soup.find_all("div", {"class": "o-teaser__heading"})

            dates = []
            headlines = []

            for elem in headline_list:
                titles = elem.findAll(text=True)
                titles_string = ''.join(titles)
                headlines.append(titles_string)

            for elem in headline_list:
                dates.append(Date)

            News = pd.DataFrame({'Date': dates, 'Headline': headlines})
            PD_Headers = pd.concat([PD_Headers, News])

PD_Headers.drop_duplicates(['Headline'], keep='last')
PD_Headers = PD_Headers[PD_Headers['Headline'].str.contains('|'.join(filter_keywords), case=False)]
PD_Headers.to_csv('files/'+query_list[0]+'_News_FT.csv')

# Combining ALL News Files into ONE
BS_data = pd.read_csv('files/'+query_list[0]+'_News_BS.csv')
FT_data = pd.read_csv('files/'+query_list[0]+'_News_FT.csv')
NYTimes_data = pd.read_csv('files/'+query_list[0]+'_News_NYTimes.csv')

BS_data = BS_data.assign(Source='BS')
BS_data = BS_data[['Date', 'Headline', 'Source']]

FT_data = FT_data.assign(Source='FT')
FT_data = FT_data[['Date', 'Headline', 'Source']]

NYTimes_data = NYTimes_data.assign(Source='NYTimes')
NYTimes_data = NYTimes_data[['Date', 'Headline', 'Source']]

all_data = pd.concat([FT_data, NYTimes_data, BS_data])
all_data.loc[:, 'Date'] = pd.to_datetime(all_data['Date'], format='%m/%d/%Y')
all_data.drop_duplicates(['Headline'], keep='last')
all_data.to_csv('files/'+query_list[0]+'_News_All.csv')

# Histogram showing Headlines per Source
AX = all_data['Source'].value_counts().plot(kind='bar')
AX.set_xlabel('Source')
AX.set_ylabel('Frequency')