#!/usr/bin/env python1.5
#!/bin/env python1.5
# $Id: wwwindex.py,v 1.8 1999/01/27 18:36:42 balay Exp balay $ 
#
# Reads in all the generated manual pages, and Creates the index
# for the manualpages, ordering the indices into sections based
# on the 'Level of Difficulty'
#
#  Usage:
#    wwwindex.py PETSC_DIR
#
import os
import posixpath
from exceptions import *
from sys import *
from string import *

# Now use the level info, and print a html formatted index
# table. Can also provide a header file, whose contents are
# first copied over.
def printindex(outfilename,headfilename,titles,tables):
      # Read in the header file
      headbuf = ''
      if posixpath.exists(headfilename) :
            try:
                  fd = open(headfilename,'r')
            except:
                  print 'Error reading file',headfilename
                  exit()
            headbuf = fd.read()
            fd.close()

      # Now open the output file.
      try:
            fd = open(outfilename,'w')
      except:
            print 'Error writing to file',outfilename
            exit()

      # Add the HTML Header info here.
      fd.write(headbuf)
      fd.write('<TABLE>')
      for i in range(len(titles)):
            title = titles[i]
            fd.write('</TR><TD>')
            fd.write('<B>' + upper(title[0])+title[1:] + '</B>')
            fd.write('</TD></TR>')
            for filename in tables[i]:
                  path,name     = posixpath.split(filename)
                  func_name,ext = posixpath.splitext(name)
                  rel_dir       = split(path,'/')[-1]
                  mesg          = '<TD WIDTH=250><A HREF="' + rel_dir + '/' + name + '">' + \
                                  func_name + '</A></TD>'
                  fd.write(mesg)
                  if tables[i].index(filename) % 3 == 2 : fd.write('<TR>')
      fd.write('</TABLE>')
      # Add HTML tail info here
      fd.write('<BR><A HREF="manualpages.html"><IMG SRC="up.xbm">Table of Contents</A>')
      fd.close()

# Read in the filename contents, and search for the formatted
# String 'Level:' and return the level info.
# Also adds the BOLD HTML format to Level field
def modifylevel(filename):
      import re
      try:
            fd = open(filename,'r')
      except:
            print 'Error! Cannot open file:',filename
            exit()
      buf    = fd.read()
      fd.close()
      re_level = re.compile(r'(Level:)\s+(\w+)')
      m = re_level.search(buf)
      level = 'none'
      if m:
            level = m.group(2)
      else:
            print 'Error! No level info in file:', filename

      # Now takeout the level info, and move it to the end,
      # and also add the bold format.
      tmpbuf = re_level.sub('',buf)
      re_loc = re.compile('(<B>Location:</B>)')
      outbuf = re_loc.sub('<B>Level:</B>' + level + r'\n<BR>\1',tmpbuf)
      
      # write the modified manpage
      try:
            #fd = open(filename[:-1],'w')
            fd = open(filename,'w')
      except:
            print 'Error! Cannot write to file:',filename
            exit()            
      fd.write(outbuf)
      fd.close()
      return level
      
# Go through each manpage file, present in dirname,
# and create and return a table for it, wrt levels specified.
def createtable(dirname,levels):
      fd = os.popen('ls '+ dirname + '/*.html')
      buf = fd.read()
      if buf == '':
            print 'Error! Empty directory:',dirname
            return None

      table = []
      for level in levels: table.append([])
      
      for filename in split(strip(buf),'\n'):
            level = modifylevel(filename)
            #if not level: continue
            if level in levels:
                  table[levels.index(level)].append(filename)
            else:
                  print 'Error! Unknown level \''+ level + '\' in', filename
      return table
      
# Gets the list of man* dirs present in the doc dir.
# Each dir will have an index created for it.
def getallmandirs(dirs):
      mandirs = []
      for filename in dirs:
            if posixpath.isdir(filename):
                  mandirs.append(filename)
      return mandirs

# Extracts PETSC_DIR from the command line and
# starts genrating index for all the manpages.
def main():
      arg_len = len(argv)
      
      if arg_len < 2: 
            print 'Error! Insufficient arguments.'
            print 'Usage:', argv[0], 'PETSC_DIR'
            exit()

      PETSC_DIR = argv[1]
      fd        = os.popen('ls -d '+ PETSC_DIR + '/docs/manualpages/man*')
      buf       = fd.read()
      dirs      = split(strip(buf),'\n')
      mandirs   = getallmandirs(dirs)

      levels = ['beginner','intermediate','advanced','developer','none']
      titles = ['Beginner: basic options',
                'Intermediate: algorithmic customization',
                'Advanced: more complex customization, including user-provided algorithms',
                'Developer: primarily for developers',
                'None: Not yet cataloged']
      for dirname in mandirs:
            table       = createtable(dirname,levels)
            if not table: continue
            outfilename    = dirname + '.html'
            dname,fname  = posixpath.split(dirname)
            headfilename = dname + '/sec/' + fname + '.head'
            printindex(outfilename,headfilename,levels,table)


# The classes in this file can also
# be used in other python-programs by using 'import'
if __name__ ==  '__main__': 
      main()
    
