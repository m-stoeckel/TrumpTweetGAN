import html
import json
import os
import re
from typing import List
from urllib import request
import matplotlib.pyplot as plt

import numpy as np
from torch.utils.data.dataset import Dataset, ConcatDataset
from tqdm import trange

handle_regex = re.compile(r'(@\w{1,15})')
hashtag_regex = re.compile(r'(#[\w_\d]+)')
dot_reply_regex = re.compile(r'^\s*\.\s*@\w{1,15}.*')
url_regex = re.compile(
    # r'((?:(?:https?|ftp)://)(?:\S+(?::\S*)?@)?(?:(?!10(?:\.\d{1,3}){3})(?!127(?:\.\d{1,3}){3})(?!169\.254(?:\.\d{1,3}){2})(?!192\.168(?:\.\d{1,3}){2})(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))|(?:(?:[a-z\x{00a1}-\x{ffff}0-9]+-?)*[a-z\x{00a1}-\x{ffff}0-9]+)(?:\.(?:[a-z\x{00a1}-\x{ffff}0-9]+-?)*[a-z\x{00a1}-\x{ffff}0-9]+)*(?:\.(?:[a-z\x{00a1}-\x{ffff}]{2,})))(?::\d{2,5})?(?:/[^\s]*)?)'
    r'((?:http[s]?|www)\S+)'
)
masked_only_regex = re.compile(
    r'^\s*(?:\s?\[[^]]+\]\s?)*\s*$'
)
clean_regex = re.compile(r'[^\w\-_!?:.,;()\d\'"\s#@]')

ALL_ACCOUNTS = (
    'ajitpaifcc', 'andypuzder', 'scaramucci', 'realbencarson', 'citizens_united', 'clewandowski_', 'sendancoats',
    'danscavino', 'darrellissa', 'sheriffclarke', 'david_bossie', 'realdonaldtrump', 'donaldjtrumpjr', 'potus',
    'erictrump', 'hillaryclinton', 'ivankatrump', 'jasoninthehouse', 'jasonmillerindc', 'senatorsessions',
    'senjohnmccain', 'kellyannepolls', 'larry_kudlow', 'linda_mcmahon', 'lindseygrahamsc', 'marcorubio',
    'michaelcohen212', 'genflynn', 'repmickmulvaney', 'govpencein', 'vp', 'repmikepompeo', 'repmikerogers',
    'senatemajldr', 'monicacrowley', 'nikkihaley', 'govhaleysc', 'pambondi', 'speakerryan', 'petehoekstra', 'reince',
    'governorperry', 'rogerjstonejr', 'vitielloronald', 'ryanzinke', 'shsanders45', 'whignewtons', 'sarahpalinusa',
    'agscottpruitt', 'scottpruittok', 'seanhannity', 'seanspicer', 'presssec', 'sebgorka', 'stephenbannon', 'tedcruz',
    'whitehouse', 'coburnforsenate', 'reptomprice'
)


# File example (url): http://trumptwitterarchive.com/data/realdonaldtrump/2019.json

# TODO: Make Iterable Over Persons (other than trump)?
class TweetDataset(Dataset):
    def __init__(self, save_dir=None, download=True, years=range(2009, 2020), account='realdonaldtrump',
                 remove_retweets=True, remove_replies=True, remove_threads=True, remove_dot_replies=False,
                 mask_handles=True, mask_hashtags=True, mask_urls=True, remove_masked_only=True, to_lower=False,
                 min_length=-1, min_count=-1, plot_tf=False):
        super(TweetDataset, self).__init__()
        if account not in ALL_ACCOUNTS:
            raise ValueError(f'"{account}" is not a valid account name:\n{ALL_ACCOUNTS}')

        self.mask_handles = mask_handles
        self.mask_hashtags = mask_hashtags
        self.mask_urls = mask_urls
        self.to_lower = to_lower
        self.min_count = min_count
        self.vocabulary = None

        file_list = []

        for year in years:
            file_list.append(str(year) + ".json")

        if save_dir is None:
            save_dir = os.path.join(os.getcwd(), 'corpus')
        account_dir = os.path.join(save_dir, account)

        if download:
            os.makedirs(account_dir, exist_ok=True)

            url_root = f'http://trumptwitterarchive.com/data/{account}/'
            for file in file_list:
                file_path = os.path.join(account_dir, file)
                if os.path.exists(file_path):
                    continue
                print(f'Downloading {file} to {account_dir}..')
                request.urlretrieve(url_root + file, file_path)

        # join tweets from all selected years in a single list
        self.tweets = []
        for file in file_list:
            file_path = os.path.join(account_dir, file)
            if os.stat(file_path).st_size == 0:
                continue
            with open(file_path, 'r', encoding='utf-8') as fp:
                try:
                    jsn = json.load(fp)
                    self.tweets.extend(jsn)
                except json.decoder.JSONDecodeError:
                    pass
        print(f'{account}: {len(self.tweets)} raw tweets')

        if remove_retweets:
            self.tweets = list(
                filter(lambda tw: not tw.get('is_retweet', True),
                       self.tweets))
        if remove_replies:
            self.tweets = list(
                filter(lambda tw: not tw.get('in_reply_to_user_id_str', None),
                       self.tweets))
        if remove_threads:
            self.tweets = list(
                filter(lambda tw: not (tw.get('text', "").endswith("..") or tw.get('text', "").startswith("..")),
                       self.tweets))
        if remove_dot_replies:
            self.tweets = list(
                filter(lambda tw: dot_reply_regex.fullmatch(tw.get('text', "")) is None,
                       self.tweets)
            )

        # Pre-process text
        self.tweets = list(map(self.pre_process, self.tweets))

        if remove_masked_only:
            self.tweets = list(
                filter(lambda tw: masked_only_regex.fullmatch(tw) is None,
                       self.tweets))
        if min_length > 0:
            self.tweets = list(
                filter(lambda tw: len(tw.split()) >= min_length,
                       self.tweets)
            )
        if self.min_count > 0:
            flat = np.array([item for sublist in map(lambda tw: tw.split(), self.tweets) for item in sublist])
            uniques, counts = np.unique(flat, return_counts=True)
            mp = {tk: ct for tk, ct in zip(uniques, counts)}
            self.vocabulary = {token for token, _ in filter(lambda tp: tp[1] >= self.min_count, zip(uniques, counts))}
            self.tweets = list(
                filter(lambda tw: len(tw) > 0,
                       map(lambda text: " ".join(filter(
                           lambda t: t in self.vocabulary, text.split())),
                           self.tweets)
                       )
            )

            if plot_tf:
                plt.title("Word Frequencies in @" + account)
                plt.ylabel("Total Number of Occurrences")
                plt.xlabel("Rank of word")
                plt.yscale('log')
                plt.plot(list(range(len(uniques))), list(sorted(counts, reverse=True)))
                plt.plot(list(range(len(self.vocabulary))),
                         list(sorted({your_key: mp[your_key] for your_key in self.vocabulary}.values(), reverse=True)))
                plt.show()

                plt.title("Word Frequencies in @" + account)
                plt.ylabel("Total Number of Occurrences")
                plt.xlabel("Rank of word")
                plt.loglog(
                    list(range(len(uniques))),
                    list(sorted(counts, reverse=True)),
                    basex=10
                )
                plt.loglog(
                    list(range(len(self.vocabulary))),
                    list(sorted({your_key: mp[your_key] for your_key in self.vocabulary}.values(), reverse=True)),
                    basex=10
                )
                plt.show()

        print(f'{account}: {len(self.tweets)} clean tweets')

    def pre_process(self, tweet):
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
        if self.mask_handles:
            text = handle_regex.sub(' [HANDLE] ', text)
        # replace hashtags with masks
        if self.mask_hashtags:
            text = hashtag_regex.sub(' [HASHTAG] ', text)
        # replace links
        if self.mask_urls:
            text = url_regex.sub(' [URL] ', text)
        text = self.tokenize(text)
        # replace spaces again
        text = re.sub(r'\s+', ' ', text)
        # lower case
        if self.to_lower:
            text = text.lower()

        return text

    @staticmethod
    def tokenize(text):
        return " ".join(re.split(r'([^\]\[\w@#\-]+|[\-]{2,})', text))

    def shuffle(self):
        np.random.shuffle(self.tweets)

    def __getitem__(self, item):
        return self.tweets[item]

    def __len__(self):
        return len(self.tweets)


min_length = 20
for year in (2009, 2015, 2016):
    data = TweetDataset(download=True, years=range(year, 2020), min_length=min_length, min_count=50)
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
        print('', flush=True, end='')
        print(f'{year}-2019 train vocabulary size: {len(vocabulary)}', flush=True)
        print(f'{year}-2019 train vocabulary[:100]: {list(vocabulary)[:100]}', flush=True)

accs = set(ALL_ACCOUNTS)
accs.remove('realdonaldtrump')
for year in (2009, 2015, 2016):
    datasets: List[TweetDataset] = []
    for acc in sorted(accs):
        dataset = TweetDataset(download=True, years=range(year, 2020), account=acc, min_length=min_length)
        np.random.seed(2020)
        dataset.shuffle()
        datasets.append(dataset)

    data = ConcatDataset(datasets)
    prefix = f'other_tweets_{year}-2019'
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
        print('', flush=True, end='')
        print(f'{year}-2019 train vocabulary size: {len(vocabulary)}', flush=True)
        print(f'{year}-2019 train vocabulary[:100]: {list(vocabulary)[:100]}', flush=True)
