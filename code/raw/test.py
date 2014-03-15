import struct
import sys

value = len("hello world")
data = struct.pack('>Q', value)
sys.stdout.write(chr(0x55)+data)
sys.stdout.write(chr(0x55)+data)