import sys
sys.path.insert(0, ".")
from logadempirical.logparser import Drain

# log_format phải khớp cấu trúc từng dòng của BGL.log:
# - 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected
log_format = "<Label> <Id> <Date> <Code1> <Time> <Code2> <Component1> <Component2> <Level> <Content>"

parser = Drain.LogParser(
    log_format,
    indir="dataset/bgl/",
    outdir="dataset/bgl/",
    depth=3,
    st=0.3,
    rex=[],
    keep_para=False,
    maxChild=100,
)
parser.parse("BGL.log")
print("DONE PARSING")
