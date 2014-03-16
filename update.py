#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
import json
from stat import S_ISREG, ST_CTIME, ST_MODE
import os, sys, time
import codecs

from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter

from jinja2 import Environment, FileSystemLoader

settings_file = open("settings.json", "r")
settings = json.loads(settings_file.read())
settings_file.close()

template_data = {}
dirpath = settings["raw_code_dir"]

entries = (os.path.join(dirpath, fn) for fn in os.listdir(dirpath))
entries = ((os.stat(path), path) for path in entries)

entries = ((stat[ST_CTIME], path) for stat, path in entries if S_ISREG(stat[ST_MODE]))

for cdate, path in sorted(entries):
    #print time.ctime(cdate), os.path.basename(path)
    extension_name = (os.path.basename(path)).split(".")[1:][0]
    if extension_name in settings["filetypes"]:
        print os.path.basename(path)
        code_file = open(path, "r")
        code = code_file.read()
        code_file.close()
        
        lexer = get_lexer_by_name(settings["filetypes"][extension_name], stripall=True)
        formatter = HtmlFormatter(linenos=True, cssclass="source", encoding="utf-8")
        
        template_data["title"] = os.path.basename(path)
        template_data["raw_link"] = "raw/" + os.path.basename(path)
        template_data["code_style"] = formatter.get_style_defs(".source")
        template_data["html_code"] = highlight(code, lexer, formatter)
        template_data["page_url"] = settings["website_url"] + settings["highlighted_code_dir"] + os.path.basename(path)+".html"
        template_data["twitter_card_title"] = os.path.basename(path) 

        env = Environment(loader=FileSystemLoader(settings["templates_dir"]))
        template = env.get_template('code_template.html')
        try:
            output_file = open(settings["highlighted_code_dir"] + os.path.basename(path)+".html", "wb")
            output_file.write(template.render(template_data).encode("ascii"))
            output_file.close()
        except UnicodeDecodeError:
            pass

