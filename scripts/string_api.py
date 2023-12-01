#!/usr/bin/env python3

################################################################
## For each protein in the given list print the names of
## their 5 best interaction partners.
##
## Requires requests module:
## type "python -m pip install requests" in command line (win)
## or terminal (mac/linux) to install the module
################################################################

import pandas as pd
import requests ## python -m pip install requests

string_api_url = "https://version-11-5.string-db.org/api"
output_format = "tsv-no-header"
method = "interaction_partners"

my_genes = ["SPP1"]

##
## Construct the request
##

request_url = "/".join([string_api_url, output_format, method])

##
## Set parameters
##

params = {

    "identifiers" : "%0d".join(my_genes), # your protein
    "species" : 9606, # species NCBI identifier 
    "limit" : 100,
    "caller_identity" : "www.awesome_app.org" # your app name

}


##
## Call STRING
##

response = requests.post(request_url, data=params)
response.text.split("\n")
##
## Read and parse the results
##
interactions=[]
for line in response.text.strip().split("\n"):

    l = line.strip().split("\t")
    query_ensp = l[0]
    query_name = l[2]
    partner_ensp = l[1]
    partner_name = l[3]
    combined_score = l[5]

    ## print

    print("\t".join([query_ensp, query_name, partner_name, combined_score]))
    interaction=[query_ensp, query_name, partner_name, combined_score]
    interactions.append(interaction)

df= pd.DataFrame(interactions, columns=["query_ensp","query_name", "partner_name", "combined_score"])
df.to_csv("interactions.tsv")