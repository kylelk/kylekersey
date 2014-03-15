from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom
import os

root = Element("root")

root = Element('root')

for root, dirs, files in os.walk("."):
    for name in dirs:
        print(os.path.join(root, name))
    for name in files:
        print(os.path.join(root, name))

#xml.dom.minidom.parseString(xml_string)
#pretty_xml_as_string = xml.toprettyxml()
#print pretty_xml_as_string
