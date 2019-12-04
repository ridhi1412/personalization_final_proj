'''format a file'''
from yapf.yapflib.yapf_api import FormatFile  # reformat a file
from glob import glob

for file in glob('*.py'):
    FormatFile(file, in_place=True)
#FormatFile('bval_muni_ndte_loader1.py', in_place=True)
#FormatFile('bval_muni_ndte_calculate_volatility3.py', in_place=True)
#FormatFile('bval_muni_ndte_calculate_error4.py', in_place=True)
#FormatFile('bval_muni_ndte_plot_graphs_5.py', in_place=True)
#FormatFile('bep_download_1a.py', in_place=True)
