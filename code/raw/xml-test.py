from xml.etree import ElementTree 
from xml.etree.ElementTree import Element 
from xml.etree.ElementTree import SubElement
import xml.dom.minidom
import time
import urllib
from BeautifulSoup import BeautifulSoup
import hashlib
import datetime
def add(item):
    localtime = time.asctime( time.localtime(time.time()))

    page_url = item
    page_data = urllib.urlopen(page_url)
    meta = page_data.info()
    page_size = meta.getheaders("Content-Length")
    page_size = ''.join(page_size)
    page_data=page_data.read()
    soup = BeautifulSoup(page_data) 
    page_title = soup.title.string
    page_digest = hashlib.sha256(page_data).hexdigest()

    # <pages/> 
    pages = Element('pages')
    
    # <pages><page/> 
    page = SubElement(pages, 'page', url=page_url)

    SubElement(page, 'info', title=page_title, size=page_size, digest=page_digest, date=localtime)

    output_file = open('pages.xml', 'w') 
    output_file.write('<?xml version="1.0"?>') 
    output_file.write(ElementTree.tostring(pages))
    output_file.close()

lines = [line.strip() for line in open('list.txt')]
for a in lines:
    add(a)