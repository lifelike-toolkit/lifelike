import requests
from bs4 import BeautifulSoup

def get_movie_urls():
    """
    Returns a dictionary of all movie titles and urls
    """
    response = requests.get('https://imsdb.com/all-scripts.html')
    soup = BeautifulSoup(response.content, 'html.parser')
    movie_urls = {}
    for link in soup.find_all('a'):
        href = link.get('href')
        if href.startswith('/Movie Scripts/'):
            movie_urls[link.get_text()] = 'https://imsdb.com' + href
    return movie_urls


def select_movie(title):
    """
    Returns the url of a selected movie
    """
    movie_urls = get_movie_urls()
    return movie_urls[title]


def get_transcript_url(movie_url):
    """
    Returns the url of the transcript of the movie
    """
    response = requests.get(movie_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    for link in soup.find_all('a'):
        href = link.get('href')
        if href.startswith('/scripts/'):
            return 'https://imsdb.com' + href


def write_transcript_to_file(url, title):
    """
    Writes the transcript to a file with the title of the movie
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    with open(title + '.txt', 'w') as f:
        f.write(soup.find('pre').get_text())


if __name__ == "__main__":
    movie_url = select_movie('Zootopia')
    transcript_url = get_transcript_url(movie_url)
    write_transcript_to_file(transcript_url, 'Zootopia')