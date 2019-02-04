#!/usr/bin/env python
"""%prog [options] [EXTENSION OUTFILE]

Create png files and a html gallery from all *.EXTENSION files in OUTFILE.
EXTENSION and OUTFILE can be omitted. They default to `pdf` and
`plots.html`, respectively.
"""

from optparse import OptionParser
import os
import subprocess
from pipes import quote

optp = OptionParser(usage=__doc__)
optp.add_option('-c', "--columns", action="store", type="int",
                help="# of columns of the gallery [default: %default]",
                default=1)
optp.add_option('-f', "--filetype", action="store", type="str",
        help="filetype of output images [default: %default]", default='png')
optp.add_option('-s', "--subdir", action="store", type="string",
                help="directory, where the images are to be found",
                default='.')
optp.add_option("--convert", action="store", type="string",
                help="ImageMagick convert command line with options, used to"
                     " create thumbnails [default: %default]",
                default="convert")

opts, args = optp.parse_args()
if len(args) == 2:
    ext = args[0]
    outpath = args[1]
elif len(args) == 0:
    ext = "pdf"
    outpath = "plots.html"
else:
    optp.error("Zero or two arguments are required!")

if not opts.convert.startswith("convert"):
    opts.convert = "convert " + opts.convert

out = open(outpath, "w")

files = filter(lambda s: s.endswith(ext), os.listdir(opts.subdir))
files.sort()

out.write('<html>\n'
          '<table>\n'
          '<tr>\n')

#newwidth = str(200./opts.columns) + '\%'

for i, f in enumerate(files):
    if i % opts.columns == 0 and i > 0:
        out.write('</tr>\n<tr>\n')
    thumb = os.path.join(opts.subdir, "tn_" + f.replace(ext, opts.filetype))
    if (not os.path.exists(thumb) or
            (os.path.getmtime(f) > os.path.getmtime(thumb))):
        cmdline = opts.convert + ' ' + quote(f) + ' ' + quote(thumb)
        # cmdline = cmdline.replace('(', '\\(').replace(')', '\\)').replace('>', '\\>')
        print cmdline
        # os.system(cmdline)
        subprocess.call(cmdline, shell=True)
    else:
        print thumb, "up to date"
    linktext = f[:-4]
    if len(linktext) > 30:
        linktext = "..." + linktext[-27:]
    out.write('<td><a href="%s">%s:<br><img src="%s"></a></td>\n' % (
              f, linktext, thumb))

out.write('</tr>\n'
          '</table>\n'
          '</html>')
