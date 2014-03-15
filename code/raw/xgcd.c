/* \begin{verbatim}
/* -------------------------------------------------------------

   Module: 	xgcd.c

   Description:

        Extended Euclidean Algorithm
        finding the greatest common divisor

        Input:  a, b    integers

        Output: x*a + y*b = d


   $Date: 2012-12-02 16:28:11 +0100 (Sun, 02 Dec 2012) $
   $Revision: 316 $

   ---------------------------------------------------------------------

   Copyright (C) 2002-2010 Berndt E. Schwerdtfeger

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   -------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>

int  main(void);

int  main(void)

{

   long u, v, u0, v0, u1, v1, u2, v2, t0, t1, t2;
   long q;

   /* get the input */

   scanf("%d %d", &u, &v);


   u0 = u; v0 = v;      	/* the following matrix is the inverse ..*/
   u1 = 1; v1 = 0;      	/* .. of my rexx procedure gcd           */
   u2 = 0; v2 = 1;              /* (u,v) * mat(u1,v1 // u2,v2) = (u0,v0) */
                                /* u1*u + u2*v = u0 = gcd(u,v)           */
   while (v0) {                 /* u = u0*|v2| , v = u0*|v1|             */

      q = u0 / v0;

      t0 = u0 - q*v0;
      t1 = u1 - q*v1;
      t2 = u2 - q*v2;

      u0 = v0;
      u1 = v1;
      u2 = v2;

      v0 = t0;
      v1 = t1;
      v2 = t2;

   }


   printf("gcd(%d,%d) = %d\n",u,v,u0);
   printf("The coefficients in x*u + y*v = d are: x = %d, y = %d\n", u1, u2);
   printf(".. the divisor are u/d = %d, v/d = %d\n", abs(v2), abs(v1));

   return 0;
}

/* \end{verbatim} */
