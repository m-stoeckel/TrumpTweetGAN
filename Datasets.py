import html
import json
import os
import re
from urllib import request

import numpy as np
from torch.utils.data.dataset import Dataset
from tqdm import trange

handle_regex = re.compile(r'(@\w{1,15})')
hashtag_regex = re.compile(r'(#[\w_\d]+)')
url_regex = re.compile(
    # r'((?:(?:https?|ftp)://)(?:\S+(?::\S*)?@)?(?:(?!10(?:\.\d{1,3}){3})(?!127(?:\.\d{1,3}){3})(?!169\.254(?:\.\d{1,3}){2})(?!192\.168(?:\.\d{1,3}){2})(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))|(?:(?:[a-z\x{00a1}-\x{ffff}0-9]+-?)*[a-z\x{00a1}-\x{ffff}0-9]+)(?:\.(?:[a-z\x{00a1}-\x{ffff}0-9]+-?)*[a-z\x{00a1}-\x{ffff}0-9]+)*(?:\.(?:[a-z\x{00a1}-\x{ffff}]{2,})))(?::\d{2,5})?(?:/[^\s]*)?)'
    r'((?:http[s]?|www)\S+)'
)
clean_regex = re.compile(r'[^\w\-_!?:.,;()\d\'"\s#@]')

REPLACE_HASHTAGS = True
REPLACE_URLS = True
REPLACE_HANDLES = False


# File example (url): http://trumptwitterarchive.com/data/realdonaldtrump/2019.json

# TODO: Make Iterable Over Persons (other than trump)?
class TrumpTweetDataset(Dataset):
    def __init__(self, save_dir=None, download=True, years=range(2009, 2020)):
        super(TrumpTweetDataset, self).__init__()
        file_list = []

        for year in years:
            file_list.append(str(year) + ".json")

        if save_dir is None:
            save_dir = os.path.join(os.getcwd(), 'corpus')

        if download:
            url_root = 'http://trumptwitterarchive.com/data/realdonaldtrump/'

            for file in file_list:
                file_path = os.path.join(save_dir, file)
                if os.path.exists(file_path):
                    continue
                print(f'Downloading {file} to {save_dir}..')
                request.urlretrieve(url_root + file, file_path)

        # join tweets from all selected years in a single list
        self.tweets = []
        for file in file_list:
            with open(os.path.join(save_dir, file), 'r', encoding='utf-8') as fp:
                jsn = json.load(fp)
                self.tweets.extend(jsn)
        print(f'loaded {len(self.tweets)} raw tweets')

        # remove re-tweets
        self.tweets = list(
            filter(lambda tw: not tw.get('is_retweet', True),
                   self.tweets))

        # remove chain tweets
        self.tweets = list(
            filter(lambda tw: not (tw.get('text', "").endswith("..") or tw.get('text', "").startswith("..")),
                   self.tweets))
        print(f'loaded {len(self.tweets)} clean tweets')

    def __getitem__(self, item):
        tweet: dict = self.tweets[item]
        text: str = tweet.get('text', "")

        # unescape HTML content
        text = html.unescape(text)

        # strip leading and trailing quotes
        text = text.strip().lstrip("\"'").rstrip("\"'")

        # replace spaces
        text = re.sub(r'\s+', ' ', text)

        # replace non text
        text = clean_regex.sub('', text)

        # replace @-handles with masks
        if REPLACE_HANDLES:
            text = handle_regex.sub(' [HANDLE] ', text)

        # replace hashtags with masks
        if REPLACE_HASHTAGS:
            text = hashtag_regex.sub(' [HASHTAG] ', text)

        # replace links
        if REPLACE_URLS:
            text = url_regex.sub(' [URL] ', text)

        text = self.tokenize(text)

        # replace spaces again
        text = re.sub(r'\s+', ' ', text)

        return text

    @staticmethod
    def tokenize(text):
        return " ".join(re.split(r'([^\]\[\w@#\-]+)', text))

    def __len__(self):
        return len(self.tweets)

    def shuffle(self):
        np.random.shuffle(self.tweets)


for year in (2009, 2015, 2016):
    data = TrumpTweetDataset(download=True, years=range(year, 2020))
    np.random.seed(2020)
    data.shuffle()
    prefix = f'trump_tweets_{year}-2019'
    with open('dataset/' + prefix + '.txt', 'w', encoding='utf-8') as train, \
            open('dataset/testdata/' + prefix + '_test.txt', 'w', encoding='utf-8') as test:
        vocabulary = set()
        train_sample = int(len(data) * 0.9)
        for i in trange(train_sample):
            entry = data[i]
            train.write(entry + "\n")
            vocabulary |= set(entry.strip().split())
        for i in trange(train_sample, len(data)):
            entry = data[i]
            test.write(entry + "\n")
            # vocabulary |= set(entry.strip().split())
        print('', flush=True, end='')
        print(f'{year}-2019 vocabulary size: {len(vocabulary)}')
        print(f'{year}-2019 vocabulary: {list(vocabulary)[:100]}')
