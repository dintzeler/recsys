# =============================================================================
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'recsys.settings')
import django

django.setup()
# import math
import ijson
import json 

import time

from recommender.models import  Venues, Papers, Log





import multiprocessing

import ijson
import tqdm



def get_or_create(model, **kwargs):
    # Actual Django statement:
    return model.objects.get_or_create(**kwargs)
    


def create(model, **kwargs):
    # Actual Django statement:
    return model.objects.create(**kwargs)


def parse(paper):
    if ("venue" in paper and "title" in paper and "id" in paper and "year" in paper and "authors" in paper):
        venue_name = paper["venue"]["raw"]
        venue, _ = get_or_create(Venues, venue_raw=venue_name)
        paper_obj = create(Papers, paper_id=paper["id"], paper_title=paper["title"], paper_year=paper["year"], venue=venue)
        for author in paper["authors"]:
            create(Log, user_id=author["id"], venue_id = venue.id, year = paper["year"], number_of_times = 1)
               
def main():
    tic = time.perf_counter()
    filename = "D:/finaldata/dblp.v12.json"
    with multiprocessing.Pool() as p, open(filename, encoding="UTF-8") as infile:
        for result in tqdm.tqdm(p.imap_unordered(parse, ijson.items(infile, "item"), chunksize=64)):
            pass
        
    toc  = time.perf_counter()
    print(toc-tic)    


if __name__ == "__main__":
    main()
