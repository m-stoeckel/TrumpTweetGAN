import json
import os
import re
from urllib import request

import numpy as np
from torch.utils.data.dataset import Dataset
from tqdm import trange

at_regex = re.compile(r'(@\w{1,15})')
hash_regex = re.compile(r'(#[\w_\d]+)')
url_regex = re.compile(
    # r'((?:(?:https?|ftp)://)(?:\S+(?::\S*)?@)?(?:(?!10(?:\.\d{1,3}){3})(?!127(?:\.\d{1,3}){3})(?!169\.254(?:\.\d{1,3}){2})(?!192\.168(?:\.\d{1,3}){2})(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))|(?:(?:[a-z\x{00a1}-\x{ffff}0-9]+-?)*[a-z\x{00a1}-\x{ffff}0-9]+)(?:\.(?:[a-z\x{00a1}-\x{ffff}0-9]+-?)*[a-z\x{00a1}-\x{ffff}0-9]+)*(?:\.(?:[a-z\x{00a1}-\x{ffff}]{2,})))(?::\d{2,5})?(?:/[^\s]*)?)'
    r'((?:http[s]?|www)\S+)'
)
clean_regex = re.compile(r'[^\w\-_!?:.,;()\d\'"\s#@]')


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

        # replace spaces
        text = re.sub(r'\s+', ' ', text)

        # replace non text
        text = clean_regex.sub('', text)

        # replace @-handles with masks
        if False:  # TODO: parametrize replacers
            text = at_regex.sub('[TW-AT]', text)

        # TODO: Create hash regex replacer

        # replace links
        text = url_regex.sub('[URL]', text)

        return text

    def __len__(self):
        return len(self.tweets)

    def shuffle(self):
        np.random.shuffle(self.tweets)


for year in (2009, 2015, 2016):
    data = TrumpTweetDataset(download=True, years=range(year, 2020))
    data.shuffle()
    prefix = f'trump_tweets_{year}-2019'
    with open(prefix + '.train', 'w', encoding='utf-8') as train, \
            open(prefix + '.dev', 'w', encoding='utf-8') as dev, \
            open(prefix + '.test', 'w', encoding='utf-8') as test:
        train_sample = int(len(data) * 0.8)
        dev_sample = int(len(data) * 0.9)
        for i in trange(train_sample):
            train.write(data[i] + "\n")
        for i in trange(train_sample, dev_sample):
            dev.write(data[i] + "\n")
        for i in trange(dev_sample, len(data)):
            test.write(data[i] + "\n")
