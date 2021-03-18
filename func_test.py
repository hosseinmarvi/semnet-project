#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 11:50:22 2020

@author: sr05
"""
def my_square(x):
    a=x*x
    return a

print(my_square(5))

def my_test(x):
    y= my_square(x)
    return y

print(my_test(7))