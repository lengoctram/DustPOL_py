#! /Users/thiemhoang/stdpy/bin/python
from pylab import *
from numpy import *
import numpy as np
import scipy as scp
import scipy.interpolate

"""	
	Using 3-point interpolation (parabol interpolation for an array of any size)
	and 2-point linear extrapolation
	Input array XX can be a scalar or an 1D array of any size, output has the same size

History:
	11.12.2014: Write interpol1D using intrinsic interpolation from scipy which allows for the choice of interpolation
			  : and linear extrapolation
			  : For monotonic data or smooth data, my previous routine 3-point interpolation is as accurate as interpol1D, but it does not allow for predefine the choice of interpolation.
	(HCT): 02/08 
"""
"""
Input:
	: Grid X0 and Y0 with X0 in increasing order----------------------
	: axis: axis of Y0 along which interpolation is performed in case of 2D or 3D array Y0
	: axis=-1 for default of Python which use the last axis of Y0 for interpolation
	: axis= 0 for first axis, 1 for second axis...
"""
def interpol1D(X0, Y0, X, kindint = 'linear', axis = -1):
	#in interpol1D, kindint and axis are optional arguments. If kindint and axis are not entered, they take default values
	Y0int= scp.interpolate.interp1d(X0, Y0, kind = kindint, axis = axis)

	N0   = int(size(X0))
	NX   = int(size(X))
	Y	 = zeros(NX)
	
	if( NX > 1):
		for ixx in range(0,NX):#xrange(0,NX):
			if((X[ixx] >= min(X0)) & (X[ixx] <= max(X0))):
				Y[ixx]  = Y0int(X[ixx])
			
			elif(X[ixx] > max(X0)):
				Nout = max([X0.argmax(),1])
				# extrapolation for X out of the provided range--------------
				A	  = (Y0[Nout]-Y0[Nout-1])/(X0[Nout] - X0[Nout-1])
				Y[ixx] = (X[ixx]-X0[Nout-1])*A+ Y0[Nout-1]

			elif(X[ixx] < min(X0)):
				Nout  = max([X0.argmin(),1])
				A	  = (Y0[Nout]-Y0[Nout-1])/(X0[Nout] - X0[Nout-1])
				Y[ixx] = (X[ixx]-X0[Nout-1])*A+ Y0[Nout-1]
			else:
				print ('good interpolation data')
			#
	else:
		if((X >= min(X0)) & (X <= max(X0))):
			Y  = Y0int(X)
		elif(X > max(X0)):
			Nout = max([X0.argmax(),1])
			# extrapolation for X out of the provided range--------------
			A   = (Y0[Nout]-Y0[Nout-1])/(X0[Nout] - X0[Nout-1])
			Y	= (X-X0[Nout-1])*A+ Y0[Nout-1]
		elif(X < min(X0)):
			Nout = max([X0.argmin(),1])
			A	 = (Y0[Nout]-Y0[Nout-1])/(X0[Nout] - X0[Nout-1])
			Y	= (X-X0[Nout-1])*A+ Y0[Nout-1]
		else:
			print ('good interpolation data')
	#
	return Y
	#
#
def interpolate1D(XX0,YY0,XX):
	N		= XX0.size
	nsnew	= XX.size
	X00		= zeros(N)
	Y00		= zeros(N)
	YY		= zeros(nsnew)
	
	X00		= XX0
	Y00		= YY0
#1
	for j in range(0,nsnew):#xrange(0,nsnew):
		if(nsnew==1):
			X=XX
		else:
			X=XX[j]
		
		if((X<XX0.max()) & (X > XX0.min())):
		#10!       X lies within the range of XX0 and YY0 and can be interpolated
			DIF=abs(X-X00[0])
			IND=1
			for I in range(0,N):#xrange(0,N):
				DIFX=abs(X-X00[I])
				if(DIFX < DIF):
					DIF=DIFX
					if((X-X00[I])<=0.):
						IND=I-1
					else:
						IND=I
					#endif
				#end
			#end for

			if (IND==N-1):
				IND=N-2

			if((IND==0)|(IND==(N-2))):
				X2=X00[IND]
				X3=X00[IND+1]
				Y2=Y00[IND]
				Y3=Y00[IND+1]

				A = (Y3-Y2)/(X3 - X2)
				Y = (X-X2)*A+ Y2
			else:

				X1=X00[IND-1]
				X2=X00[IND]
				X3=X00[IND+1]

				Y1=Y00[IND-1]
				Y2=Y00[IND]
				Y3=Y00[IND+1]

				A=(Y3-Y2-(X3-X2)*(Y1-Y2)/(X1-X2))/((X3-X2)*(X3-X1))
				B=(Y1-Y2)/(X1-X2)-A*(X1-X2)

				Y=(A*(X-X2)+B)*(X-X2)+Y2
			#endif

		else:
		#X lies outside X00 range, extrapolate for the value
			if(X==X00.max()):
				IND	= size(X00)
				Y	= Y00[IND-1]
			#endif
			if(X==X00.min()):
				Y	= Y00[0]
			#endif
			
			if((X>X00.max()) | (X<X00.min())):
				if(X > X00.max()):
					IND	= size(X00)-1
				#endif
				if(X < XX0.min()):
					IND	= 1
				#endif
			
				X2=X00[IND-1]
				X3=X00[IND]
				
				Y2=Y00[IND-1]
				Y3=Y00[IND]
				
				A = (Y3-Y2)/(X3 - X2)
				Y = (X-X2)*A+ Y2
				#end extrapolate
		#endif
					
		if(nsnew!=1):
			YY[j]	= Y
		else:
			YY		= Y
		#1	endfor

	return YY

#end subroutine interpolate1D
#end module
