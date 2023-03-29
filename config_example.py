# 默认参数，不太需要修改
# Limit the minimal length of your response
RESPONSE_THRESH = 6
# the augmentation factor can be reduced if you enable the parsing of replies
AUGMENTATION_FACTOR_ORIGINAL = 2
# better just 1, as we are using full context
AUGMENTATION_FACTOR_REPLIES = 1

# 以下都是可选功能，可以根据需要开启
# More interestingly, let's ask OpenAI to make a question for your tweets
ENABLE_OPENAI = False
OPENAI_KEY = ''
OPENAI_MAX_SAMPLE = 2        # dont make it larger than the augmentation factor
OPENAI_THREADS = 128         # Be careful when you try this! It can be expensive.
OPENAI_CACHE_ONLY = False

# Parsing replies is not recommended, as it will take a lot of time
# But it can increase the overall quality significantly
PARSE_REPLIES = False
SCRAPE_TIMEOUT = 20
SCRAPE_CACHE_ONLY = False
PROXY = False
PROXY_ADDR = ""
PROXY_PORT = ""
PROXY_USER = ""
PROXY_PASSWD = ""

# numbers of processes to used for selenium
# base on experience this is better 2-3 times your CPU cores. Use it wisely.
PROCESSES = 8