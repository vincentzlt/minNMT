import re
import sys


def cjk_detect(texts):
    # korean
    if re.search("[\uac00-\ud7a3]", texts):
        return True
    # japanese
    if re.search("[\u3040-\u30ff]", texts):
        return True
    # chinese
    if re.search("[\u4e00-\u9FFF]", texts):
        return True
    return False


if __name__ == "__main__":
    for l in sys.stdin:
        if not cjk_detect(l):
            print(l.strip())