import json

class Dataset():

    def __init__(self, dataset_path, dataset_root_path, k):
        self.k = k
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        f.close()
        self.data = data
        self.iter = data.__iter__()
        with open(dataset_root_path, 'r') as f:
            root = json.load(f)
        f.close()
        self.root = root
        with open('social_network.json', 'r') as f:
            social = json.load(f)
        f.close()
        self.social = social

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        val = self.data[key]
        return self.__process_item(val, key)

    def __next__(self):
        try:
            key = self.iter.__next__()
            val = self.data[key]
            return self.__process_item(key, val)
        except StopIteration:
            raise StopIteration

    def __process_item(self, key, cascade):
        for ckey in cascade.keys():
            inner = cascade[ckey]
            # uncomment if you need the followee id's as well
            try:
                #inner["user_followees"] = self.social[inner["user"]]
                inner["user_followees_count"] = len(self.social[inner["user"]])
            except KeyError:
                #inner["user_followees"] = []
                inner["user_followees_count"] = 0
        result = {"url": key, "cascade": cascade}
        result["cascade_length"] = len(cascade)
        result["cascade_root"] = self.__cascade_root(cascade)
        result["root_tweet_id"] = self.root[key]
        result["root_tweet"] = self.__get_tweet(self.root[key])
        result["label"] = result["cascade_length"] >= 2*self.k
        return result

    def __cascade_root(self, cascade):
        tweets = cascade.keys()
        return str(sorted(map(int, tweets))[0])

    def __get_tweet(self, tweetid):
        with open('tweets/'+tweetid+'.json', 'r', encoding='utf-8') as f:
            return json.load(f)

def get_flattened_data(dataset_path, dataset_root_path, k):
    dataset = []
    for data in Dataset(dataset_path, dataset_root_path, k):
        dataset.append(data)
    return dataset

if __name__ == "__main__":
    
    # Assumes this script is in the root folder, with dataset and tweets folder from assignment2 handout
    # and social_network.json is in the root folder 
    #
    # import Tweet
    # dataset = Tweet.get_flattened_data('dataset/k2/training.json', 'dataset/k2/root_tweet.json', 2)
    #
    # dataset then contains an array of dicts with the following format
    # {
    #     "url":"http://4sq.com/7A59zZ",
    #     "cascade_length":2,
    #     "cascade_root":"1",
    #     "root_tweet_id":"25531096591",
    #     "label":False,
    #     "cascade":{
    #         "2":{
    #             "created_at":"1285540781000",
    #             "user":"47098585",
    #             "user_followees_count":7,
    #             "id":"25630185400"
    #         },
    #         "1":{
    #             "created_at":"1285448859000",
    #             "user":"14192436",
    #             "user_followees_count":5,
    #             "id":"25531096591",
    #             "in_reply_to":-1
    #         }
    #     }
    #     "root_tweet":{
    #         "is_quote_status":False,
    #         "coordinates":None,
    #         "id_str":"25531096591",
    #         "user":{
    #             "following":False,
    #             "contributors_enabled":False,
    #             "geo_enabled":True,
    #             "screen_name":"rolf_laun",
    #             "is_translator":False,
    #             "id_str":"14192436",
    #             "profile_sidebar_border_color":"829D5E",
    #             "profile_image_url":"http://pbs.twimg.com/profile_images/800375965/twitterProfilePhoto_normal.jpg",
    #             "profile_sidebar_fill_color":"99CC33",
    #             "listed_count":11,
    #             "statuses_count":2436,
    #             "profile_background_tile":False,
    #             "favourites_count":30,
    #             "protected":False,
    #             "id":14192436,
    #             "profile_link_color":"D02B55",
    #             "follow_request_sent":False,
    #             "description":"Another techie, librarian nerd playing aournd on the internet",
    #             "profile_background_image_url":"http://abs.twimg.com/images/themes/theme5/bg.gif",
    #             "created_at":"Fri Mar 21 17:18:23 +0000 2008",
    #             "time_zone":"Central Time (US & Canada)",
    #             "url":None,
    #             "is_translation_enabled":False,
    #             "friends_count":72,
    #             "location":"29.464936,-98.532377",
    #             "name":"Rolf Laun",
    #             "profile_background_color":"352726",
    #             "translator_type":"none",
    #             "profile_use_background_image":True,
    #             "profile_text_color":"3E4415",
    #             "followers_count":166,
    #             "default_profile":False,
    #             "has_extended_profile":False,
    #             "profile_background_image_url_https":"https://abs.twimg.com/images/themes/theme5/bg.gif",
    #             "utc_offset":-21600,
    #             "default_profile_image":False,
    #             "lang":"en",
    #             "entities":{
    #                 "description":{
    #                     "urls":[

    #                     ]
    #                 }
    #             },
    #             "notifications":False,
    #             "verified":False,
    #             "profile_image_url_https":"https://pbs.twimg.com/profile_images/800375965/twitterProfilePhoto_normal.jpg"
    #         },
    #         "in_reply_to_user_id_str":None,
    #         "source":"<a href="http://foursquare.com" rel="nofollow">Foursquare</a>",
    #         "truncated":False,
    #         "favorite_count":0,
    #         "in_reply_to_status_id":None,
    #         "retweet_count":0,
    #         "retweeted":False,
    #         "favorited":False,
    #         "id":25531096591,
    #         "place":None,
    #         "created_at":"Sat Sep 25 21:07:39 +0000 2010",
    #         "lang":"en",
    #         "geo":None,
    #         "in_reply_to_status_id_str":None,
    #         "in_reply_to_screen_name":None,
    #         "contributors":None,
    #         "text":"I"m at Apple Store (7400 San Pedro, San Antonio) w/ 2 others. http://4sq.com/7A59zZ",
    #         "in_reply_to_user_id":None,
    #         "entities":{
    #             "urls":[

    #             ],
    #             "symbols":[

    #             ],
    #             "hashtags":[

    #             ],
    #             "user_mentions":[

    #             ]
    #         }
    #     }
    # }