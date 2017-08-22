# 読んでわかる
import glob, re

_list = glob.glob('./natsume/*')

for idx, txt in enumerate(_list):
  origin = open (_list[idx], 'r', encoding='utf-8').read()
  with open ('{}.txt'.format(_list[idx]), 'w', encoding='utf-8') as f:
    try:
      # extract main content
      content = origin[origin.index('---\n\n')+4 : origin.index('\n底本')]
      # delete hiragana in brackets
      rm_string = re.findall(r"《.*》", content)
      for bracket in rm_string:
        content = content.replace(bracket, '')
      f.write(content)
    except ValueError as e:
      print ('now file: ', idx)
      input ('Press Enter to continue...')
