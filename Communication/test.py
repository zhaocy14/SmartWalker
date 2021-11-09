
import re

string = "1980000000 0.67 03530.00 188"
m = re.match(r"(?P<time>\d+) (?P<theta>\d+\.\d+) (?P<dist>\d+\.\d+) (?P<q>\d+)", string)
print(m.groupdict())