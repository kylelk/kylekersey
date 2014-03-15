    #
    #  divisors.py
    #  
    #
    #  Created by Kyle Kersey on 8/19/13 4:33 PM.
    #
    #


b=1
d = 0
#generates a list of numbers.
while b<100:
    b=b+1
    x = 0.0
    a = 0
    #generates a list of numbers less than b.
    while x<b:
        x=x+1
        #this will check for divisors.
        if (b/x)-int(b/x) == 0.0:
            a=a+1
    if a==2:
        print b