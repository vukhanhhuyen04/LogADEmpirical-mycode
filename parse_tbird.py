import sys
sys.path.insert(0, ".")
from logadempirical.logparser import Drain

# log_format phải khớp cấu trúc từng dòng của Thunderbird10M.log:
# - 1147122219 2006.05.08 cn505 May 8 14:03:39 cn505/cn505 kernel: NET: Registered protocol family 10
log_format = "<Label> <Id> <Date> <Admin> <Month> <Day> <Time> <AdminAddr> <Content>"

parser = Drain.LogParser(
    log_format,
    indir="dataset/tbird/",
    outdir="dataset/tbird/",
    depth=3,
    st=0.3,
    rex=[],
    keep_para=False,
    maxChild=100,
)
parser.parse("Thunderbird10M.log")
print("DONE PARSING")
