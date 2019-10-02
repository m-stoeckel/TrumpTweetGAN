#!/usr/bin/env bash
sed -i -E "s/([^][[:alnum:]\@#]+)/ \1 /g;s/^\s//;s/\s+/ /g" trump_tweets_20*