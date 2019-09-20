import re
from urllib import request
import os
from zipfile import ZipFile
from typing import *
import shutil

import numpy as np
from torch.utils.data.dataset import Dataset
import tempfile
import json

at_regex = re.compile(r'(@\w{1,15})')
hash_regex = re.compile(r'(#[\w_\d]+)')
url_regex = re.compile(
    # r'((?:(?:https?|ftp)://)(?:\S+(?::\S*)?@)?(?:(?!10(?:\.\d{1,3}){3})(?!127(?:\.\d{1,3}){3})(?!169\.254(?:\.\d{1,3}){2})(?!192\.168(?:\.\d{1,3}){2})(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))|(?:(?:[a-z\x{00a1}-\x{ffff}0-9]+-?)*[a-z\x{00a1}-\x{ffff}0-9]+)(?:\.(?:[a-z\x{00a1}-\x{ffff}0-9]+-?)*[a-z\x{00a1}-\x{ffff}0-9]+)*(?:\.(?:[a-z\x{00a1}-\x{ffff}]{2,})))(?::\d{2,5})?(?:/[^\s]*)?)'
    r'((?:http[s]?|www)\S+)'
)
clean_regex = re.compile(r'[^\w\-_!?:.,;()\d\'"\s#@]')

# TODO: Make IterableDataset?
class TrumpTweetDataset(Dataset):
    def __init__(self, save_dir=None, download=True):
        super(TrumpTweetDataset, self).__init__()
        file_list: List[str] = [  # 'condensed_2009.json.zip', 'condensed_2010.json.zip', 'condensed_2011.json.zip',
            # 'condensed_2012.json.zip', 'condensed_2013.json.zip', 'condensed_2014.json.zip',
            # 'condensed_2015.json.zip',
            'condensed_2016.json.zip', 'condensed_2017.json.zip', 'condensed_2018.json.zip']
        if save_dir is None:
            save_dir = os.path.join(os.getcwd(), 'corpus')

        if download:
            tempdir = tempfile.mkdtemp('TrumpGAN')
            url_root = 'https://github.com/bpb27/trump_tweet_data_archive/raw/master/'

            for file_name in file_list:
                print(f'Downloading {file_name} to {save_dir}..')
                file_path = os.path.join(tempdir, file_name)
                request.urlretrieve(url_root + file_name, file_path)
                with ZipFile(file_path, 'r') as zip:
                    zip.extractall(save_dir)

            # cleanup
            shutil.rmtree(tempdir)

        # join tweets from all selected years in a single list
        file_list = list(map(lambda s: s.replace('.zip', ''), file_list))
        self.tweets = []
        for file in file_list:
            with open(os.path.join(save_dir, file), 'r', encoding='utf-8') as fp:
                jsn = json.load(fp)
                self.tweets.extend(jsn)
        print(f'loaded {len(self.tweets)} raw tweets')

        # remove re-tweets
        self.tweets = list(filter(lambda tw: not tw.get('is_retweet', True), self.tweets))
        print(f'loaded {len(self.tweets)} clean tweets')

    def __getitem__(self, item):
        tweet: dict = self.tweets[item]
        text: str = tweet.get('text', "")

        # replace spaces
        text = re.sub(r'\s+', ' ', text)

        # replace non text
        text = clean_regex.sub('', text)

        # replace @-handles with masks
        if False: # TODO: parametrize replacers
            text = at_regex.sub('[TW-AT]', text)

        # TODO: Create hash regex replacer

        # replace links
        text = url_regex.sub('[URL]', text)

        return text

    def __len__(self):
        return len(self.tweets)


data = TrumpTweetDataset(download=False)
print(data[0])
