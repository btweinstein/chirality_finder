__author__ = 'bryan'

'''These files are ripped almost directly from
http://wiki.scipy.org/Cookbook/Least_Squares_Circle#head-f353234ac19e695689375c7eaa2df9ab64938a03'''

from numpy import *
from scipy import  odr

#### Utility Functions ######

def calc_R(x, y, xc, yc):
    """ calculate the distance of each 2D points from the center c=(xc, yc) """
    return sqrt((x-xc)**2 + (y-yc)**2)

#### ALGEBRAIC ##########

def algebraic_circleFind(x, y, guess = None):
    # == METHOD 1: Algebraic ==

    # coordinates of the barycenter
    if guess is None:
        x_m = mean(x)
        y_m = mean(y)
    else:
        x_m = guess[0]
        y_m = guess[1]

    # calculation of the reduced coordinates
    u = x - x_m
    v = y - y_m

    # linear system defining the center in reduced coordinates (uc, vc):
    #    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
    #    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
    Suv  = sum(u*v)
    Suu  = sum(u**2)
    Svv  = sum(v**2)
    Suuv = sum(u**2 * v)
    Suvv = sum(u * v**2)
    Suuu = sum(u**3)
    Svvv = sum(v**3)

    # Solving the linear system
    A = array([ [ Suu, Suv ], [Suv, Svv]])
    B = array([ Suuu + Suvv, Svvv + Suuv ])/2.0
    uc, vc = linalg.solve(A, B)

    xc = x_m + uc
    yc = y_m + vc

    # Calculation of all distances from the center (xc_1, yc_1)
    Ri      = sqrt((x-xc)**2 + (y-yc)**2)
    R       = mean(Ri)
    residu  = sum((Ri- R)**2)
    residu2 = sum((Ri **2-R**2)**2)

    return (xc, yc), (R, Ri), (residu, residu2)

######### LEAST SQUARES ############

def leastSq_circleFind(x, y, guess=None):

    def f_2(c):
        """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(x, y, *c)
        return Ri - Ri.mean()

    if guess is None:
        center_estimate = mean(x), mean(y)
    else:
        center_estimate = guess
    center, ier = optimize.leastsq(f_2, center_estimate)

    xc, yc = center
    Ri       = calc_R(x, y, xc, yc)
    R        = Ri.mean()
    residu   = sum((Ri - R)**2)
    residu2  = sum((Ri**2-R**2)**2)

    return (xc, yc), (R, Ri), (residu, residu2)

from scipy import optimize

def leastSq_circleFind_jacobian(x, y, guess=None):
    # == METHOD 2b ==
    # Advanced usage, with jacobian

    def f_2b(c):
        """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(x,y, *c)
        return Ri - Ri.mean()

    def Df_2b(c):
        """ Jacobian of f_2b
        The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
        xc, yc     = c
        df2b_dc    = empty((len(c), x.size))

        Ri = calc_R(x,y, xc, yc)
        df2b_dc[ 0] = (xc - x)/Ri                   # dR/dxc
        df2b_dc[ 1] = (yc - y)/Ri                   # dR/dyc
        df2b_dc       = df2b_dc - df2b_dc.mean(axis=1)[:, newaxis]

        return df2b_dc

    if guess is None:
        center_estimate = mean(x), mean(y)
    else:
        center_estimate = guess
    center_2b, ier = optimize.leastsq(f_2b, center_estimate, Dfun=Df_2b, col_deriv=True)

    xc_2b, yc_2b = center_2b
    Ri_2b        = calc_R(x, y, xc_2b, yc_2b)
    R_2b         = Ri_2b.mean()
    residu_2b    = sum((Ri_2b - R_2b)**2)
    residu2_2b   = sum((Ri_2b**2-R_2b**2)**2)

    return (xc_2b, yc_2b), (R_2b, Ri_2b), (residu_2b, residu2_2b)

####### ODR ###########

def odr_circleFind(x, y, guess=None):
    # Basic usage of odr with an implicit function definition
    def f_3(beta, x):
        """ implicit definition of the circle """
        return (x[0]-beta[0])**2 + (x[1]-beta[1])**2 -beta[2]**2

    # initial guess for parameters
    if guess is None:
        x_m, y_m = mean(x), mean(y)
    else:
        x_m, y_m = guess[0], guess[1]

    R_m = calc_R(x, y, x_m, y_m).mean()
    beta0 = [ x_m, y_m, R_m]


    # for implicit function :
    #       data.x contains both coordinates of the points
    #       data.y is the dimensionality of the response
    lsc_data   = odr.Data(row_stack([x, y]), y=1)
    lsc_model  = odr.Model(f_3, implicit=True)
    lsc_odr    = odr.ODR(lsc_data, lsc_model, beta0)
    lsc_out    = lsc_odr.run()

    xc_3, yc_3, R_3 = lsc_out.beta
    Ri_3       = calc_R(x, y,xc_3, yc_3)
    residu_3   = sum((Ri_3 - R_3)**2)
    residu2_3  = sum((Ri_3**2-R_3**2)**2)

    return (xc_3, yc_3), (R_3, Ri_3), (residu_3, residu2_3)

def odr_circleFind_jacobian(x, y):
    # Advanced usage, with jacobian

    def f_3b(beta, x):
        """ implicit definition of the circle """
        return (x[0]-beta[0])**2 + (x[1]-beta[1])**2 -beta[2]**2

    def jacb(beta, x):
        """ Jacobian function with respect to the parameters beta.
        return df_3b/dbeta
        """
        xc, yc, r = beta
        xi, yi    = x

        df_db    = empty((beta.size, x.shape[1]))
        df_db[0] =  2*(xc-xi)                     # d_f/dxc
        df_db[1] =  2*(yc-yi)                     # d_f/dyc
        df_db[2] = -2*r                           # d_f/dr

        return df_db

    def jacd(beta, x):
        """ Jacobian function with respect to the input x.
        return df_3b/dx
        """
        xc, yc, r = beta
        xi, yi    = x

        df_dx    = empty_like(x)
        df_dx[0] =  2*(xi-xc)                     # d_f/dxi
        df_dx[1] =  2*(yi-yc)                     # d_f/dyi

        return df_dx


    def calc_estimate(data):
        """ Return a first estimation on the parameter from the data  """
        xc0, yc0 = data.x.mean(axis=1)
        r0 = sqrt((data.x[0]-xc0)**2 +(data.x[1] -yc0)**2).mean()
        return xc0, yc0, r0

    # for implicit function :
    #       data.x contains both coordinates of the points
    #       data.y is the dimensionality of the response
    lsc_data  = odr.Data(row_stack([x, y]), y=1)
    lsc_model = odr.Model(f_3b, implicit=True, estimate=calc_estimate, fjacd=jacd, fjacb=jacb)
    lsc_odr   = odr.ODR(lsc_data, lsc_model)    # beta0 has been replaced by an estimate function
    lsc_odr.set_job(deriv=3)                    # use user derivatives function without checking
    lsc_odr.set_iprint(iter=1, iter_step=1)     # print details for each iteration
    lsc_out   = lsc_odr.run()

    xc_3b, yc_3b, R_3b = lsc_out.beta
    Ri_3b       = calc_R(x, y, xc_3b, yc_3b)
    residu_3b   = sum((Ri_3b - R_3b)**2)
    residu2_3b  = sum((Ri_3b**2-R_3b**2)**2)

    return (xc_3b, yc_3b), (R_3b, Ri_3b), (residu_3b, residu2_3b)