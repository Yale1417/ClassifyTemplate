import os
import requests

# proxy = {"http": "http://127.0.0.1:7890", "https": "https://127.0.0.1:7890"}


def downPic(path, url):
    if os.path.exists(path):
        return
    try:
        r = requests.get(url, timeout=30)
        with open(path, "wb") as f:
            f.write(r.content)
    except:
        print("error,", url)


# requests.get("https://www.reddit.com/r/Hegoesdown/top/?t=all")
for line in open("urls_porn.txt"):
    url = line.strip()
    name = "porn_" + url.split("/")[-1]
    print(name, url)
    downPic(name, url)
    exit()
