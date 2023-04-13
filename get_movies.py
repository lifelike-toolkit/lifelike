import requests
from bs4 import BeautifulSoup


def write_transcript_to_file(url, title):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    with open(title + '.txt', 'w') as f:
        f.write(soup.find('pre').get_text())


if __name__ == "__main__":
    url = 'https://imsdb.com/scripts/Star-Wars-A-New-Hope.html'
    title = 'Star Wars: A New Hope'
    write_transcript_to_file(url, title)