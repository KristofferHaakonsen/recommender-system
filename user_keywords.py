import json
import pandas as pd
import os


def load_content_data(path):
    article_lst = []
    extract_fields = ['id', 'publishtime', 'title', 'description', 'keyword',
                      'kw-classification', 'kw-category', 'kw-concept', 'kw-company', 'kw-entity', 'kw-location']
    for f in os.listdir(path):
        file_name = os.path.join(path, f)
        if os.path.isfile(file_name):
            obj = json.loads(open(file_name).readline().strip())
            if not obj is None:
                d = {}
                # TODO: extract correct field from list of fields
                for field in obj['fields']:
                    if field['field'] in extract_fields:
                        d[field['field']] = field['value']
                article_lst.append(d)
    return article_lst


if __name__ == '__main__':
    articles = load_content_data('data/content_refine')
    df = pd.DataFrame(articles)
    # TODO: convert time to datetime
