import os
import glob
import json
import os, sys, time
import codecs

from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter

from jinja2 import Environment, FileSystemLoader
import BeautifulSoup

import xml.etree.ElementTree as ET


settings_file = open("settings.json", "r")
settings = json.loads(settings_file.read())
settings_file.close()

template_data = {}
dirpath = settings["raw_code_dir"]

discription_template = """
    <div class="code_info">
    <h4>{file_name}</h4>
    </div>"""

for file_name in os.listdir(dirpath):
    path = dirpath+file_name
    extension_name = (os.path.basename(path)).split(".")[1:][0]
    if extension_name in settings["filetypes"]:
        code_file = open(path, "r")
        code = code_file.read()
        code_file.close()
        soup = BeautifulSoup.BeautifulSoup(code)
        doc_xml = soup.find("div", {"class": "code_info"})

        if doc_xml == None:
            template_data[file_name] = discription_template.format(file_name=file_name)
        else:
            template_data[file_name] = doc_xml

env = Environment(loader=FileSystemLoader(settings["templates_dir"]))
template = env.get_template('code_listings_template.html')


