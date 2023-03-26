# Warning: research purposes only. This code is not intended for production use. Use at your own risk. 
# Scraping Twitter may violate the Terms of Service.

import multiprocessing
import re
import threading
import zipfile
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import concurrent.futures
import json
import os
import config

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.proxy import Proxy, ProxyType
from selenium.webdriver.common.keys import Keys

class document_is_ready(object):
    def __call__(self, driver):
        ready_state = driver.execute_script("return document.readyState")
        return ready_state == "complete"

def get_tweet_info(tweet):

    # check out the cache first,
    # if hit, return the cached result
    tweet_id = tweet['id']

    try:
        with open(f'tweet_cache/{tweet_id}.txt', 'r') as f:
            return {'id': tweet_id, 'text': tweet['text'], 'context': json.load(f)}
    except:
        pass

    if config.CACHE_ONLY:
        return None

    driver = get_driver()

    try:
        url = f"https://mobile.twitter.com/ljsabc/status/{tweet_id}"
        driver.get(url)
        WebDriverWait(driver, config.SCRAPE_TIMEOUT).until(EC.presence_of_element_located((By.XPATH, '//*[@data-testid="bookmark"]')))
        
        # Wait for DOM to be ready
        wait = WebDriverWait(driver, config.SCRAPE_TIMEOUT)
        wait.until(document_is_ready())

        body_element = driver.find_element(By.TAG_NAME, "body")


        for j in range(2):
            # looks like we need to scroll up a few times to get the full context
            # 2 times for scroll up should be okay for most of the tweets
            # if you tweet longer you may try a longer range
            for j in range(3):
                body_element.send_keys(Keys.PAGE_UP)
                time.sleep(0.15)
            # introduce a small delay to let the page load
            # the delays can be hidden by using more threads
            time.sleep(1.0)

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        articles = soup.find_all('article')

        if articles is None:
            # weird, probably a deleted tweet
            # or, possibly web driver failed to load the page
            # at this moment, we just return None and let the caller handle it
            return None

        target_index = -1
        for i, article in enumerate(articles):
            if article.find(lambda tag: tag.get('data-testid') == 'bookmark'):
                target_index = i
                break

        # without scroll to top, the tweet context may not be complete
        # but as we are only targetting a small context range, it should not hurt.
        print(f"id: {tweet_id}, located {target_index + 1} of {len(articles)} tweets.")

        results = []
        if target_index >= 0:
            for i, article in enumerate(articles):
                if i < target_index:
                #f True:
                    target = article.find('div', {'data-testid': 'tweetText'})
                    if target:
                        tweet_text = target.get_text()
                        print(i, tweet_text)
                        results.append(tweet_text)

        # dump the results to a file
        with open(f'tweet_cache/{tweet_id}.txt', 'w') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"Original tweet: {tweet['text']}\n\n")

        return {'id': tweet_id, 'text': tweet['text'], 'context': results}
    except Exception as e:
        print(e)
        #return {'id': tweet_id, 'text': tweet['text'], 'context': None}
        return None
    


threadLocal = threading.local()

def get_driver():
    driver = getattr(threadLocal, 'driver', None)
    if driver is None:
        mobile_user_agent = 'Mozilla/5.0 (Linux; Android 9; Pixel 3 Build/PQ3A.190705.001) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.101 Mobile Safari/537.36'
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument(f'user-agent={mobile_user_agent}')
        prefs = {"profile.managed_default_content_settings.images": 2}
        chrome_options.add_experimental_option("prefs", prefs)        
        chrome_options.add_argument(f"user-data-dir=chrome_profile/{threading.current_thread().ident}")

        # The window is place on purpose to avoid "error request".

        if config.PROXY:
            #proxy or proxy pool, it will be really useful
            pluginfile = 'proxy_auth_plugin.zip'
            manifest_json = """
            {
                "version": "1.0.0",
                "manifest_version": 2,
                "name": "Chrome Proxy",
                "permissions": [
                    "proxy",
                    "tabs",
                    "unlimitedStorage",
                    "storage",
                    "<all_urls>",
                    "webRequest",
                    "webRequestBlocking"
                ],
                "background": {
                    "scripts": ["background.js"]
                },
                "minimum_chrome_version":"22.0.0"
            }
            """

            background_js = """
            var config = {
                    mode: "fixed_servers",
                    rules: {
                    singleProxy: {
                        scheme: "http",
                        host: "%s",
                        port: parseInt(%s)
                    },
                    bypassList: ["localhost"]
                    }
                };

            chrome.proxy.settings.set({value: config, scope: "regular"}, function() {});

            function callbackFn(details) {
                return {
                    authCredentials: {
                        username: "%s",
                        password: "%s"
                    }
                };
            }

            chrome.webRequest.onAuthRequired.addListener(
                        callbackFn,
                        {urls: ["<all_urls>"]},
                        ['blocking']
            );""" % (config.PROXY_ADDR, config.PROXY_PORT, config.PROXY_USER, config.PROXY_PASSWD)


            with zipfile.ZipFile(pluginfile, 'w') as zp:
                zp.writestr("manifest.json", manifest_json)
                zp.writestr("background.js", background_js)
            chrome_options.add_extension(pluginfile)
            driver = webdriver.Chrome(executable_path='./chromedriver', options=chrome_options)
        else:
            driver = webdriver.Chrome(executable_path='./chromedriver', options=chrome_options)
        driver.set_window_size(340, 695)
        setattr(threadLocal, 'driver', driver)

    return driver

def process_tweet_ids(tweets):
    # create a cache folder to store the scraped tweets
    if not os.path.exists('tweet_cache'):
        os.mkdir('tweet_cache')

    if not os.path.exists('chrome_profile'):
        os.mkdir('chrome_profile')
    else:
        # remove the dir, then create a new one
        # this is to avoid the cache issue
        import shutil
        shutil.rmtree('chrome_profile')
        os.mkdir('chrome_profile')

    tweet_data = []
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=config.PROCESSES)
    for result in pool.map(get_tweet_info, tweets):
        if result is not None:
            tweet_data.append(result)

    return tweet_data

if __name__ == '__main__':
    # Replace with a list of tweet IDs you want to fetch information for
    tweet_ids = ['1639865895374958592', '1639689372059725825', '1639488084705439745', '1639639351897522176']
    tweet_data = process_tweet_ids(tweet_ids)

    for data in tweet_data:
        print(data)