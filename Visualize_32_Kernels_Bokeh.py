# -*- coding: utf-8 -*-
"""
Created on Wed May 03 11:39:18 2017
    
@author: AZakaria
"""    

from bokeh.io import output_file,show
from bokeh.plotting import figure
from bokeh.layouts import row
from bokeh.models.widgets import Tabs,Panel

def weights_32_visualize(Weights_reshaped):
    p=[]
    for i in range (0,32):
        p.append(figure(x_range=(0,3),y_range=(0,3)))
        p[i].image(image=[Weights_reshaped[i]],x=0,y=0,dw=3, dh=3,palette="Greys8")
        
    panel=[]
 
    for i in range (0,8):
        panel.append(Panel(child=row(p[4*i],p[4*i+1],p[4*i+2],p[4*i+3]),title="filters"+str(4*i)+"-"+str(4*i+3)))
    

    layout=Tabs(tabs=[ panel[0], panel[1], panel[2], panel[3], panel[4], panel[5], panel[6], panel[7]])
    output_file('layer1_32filters.html')
    show(layout)
        
    
        
        
        
