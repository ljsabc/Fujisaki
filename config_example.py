# Limit the minimal length of your response
RESPONSE_THRESH = 6
# the augmentation factor can be reduced if you enable the parsing of replies
AUGMENTATION_FACTOR_ORIGINAL = 4
AUGMENTATION_FACTOR_REPLIES = 2

PARSE_REPLIES = False
# If you want to scrape only from cache, set this to True
CACHE_ONLY = False
SCRAPE_TIMEOUT = 30
PROXY = False
PROXY_ADDR = ""
PROXY_PORT = ""
PROXY_USER = ""
PROXY_PASSWD = ""

# This should be is better 2-3 times your CPU cores.
PROCESSES = 4