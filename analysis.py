from scipy.optimize import curve_fit
import numpy as np
from scipy import stats

try:
    import uncertainties.unumpy as unp
    import uncertainties as unc
except:
    try:
        from pip import main as pipmain
    except:
        from pip._internal import main as pipmain
    pipmain(['install', 'uncertainties'])
    import uncertainties.unumpy as unp
    import uncertainties as unc

x = np.array([1,2,3,4,5,6,7])
y = [2,4,5,7,8,9,11]


def fit_function(y, initPoint, x=None, forecast=0, maxfev=10000, method='trf'):
    n = len(y)
    if x is None:
        x = np.arange(1, n + 1)

    def f(x, *args):
        param = []
        for i in args:
            param.append(i)
        return param[0] * np.exp(param[1] * (x - param[2])) + param[3]

    popt, pcov = curve_fit(f, x, y, p0=initPoint, method=method, maxfev=maxfev)

    # retrieve parameter values
    a = popt[0]
    b = popt[1]
    c = popt[2]
    d = popt[3]

    # compute r^2
    r2 = 1.0-(sum((y-f(x,a,b,c,d))**2)/((n-1.0)*np.var(y,ddof=1)))

    # calculate parameter confidence interval
    a, b, c, d = unc.correlated_values(popt, pcov)

    # calculate regression confidence interval
    px = np.linspace(1, n+forecast, n+forecast)
    py = a*unp.exp(b*(px-c))+d
    nom = unp.nominal_values(py)
    std = unp.std_devs(py)

    def predband(x, xd, yd, p, func, conf=0.95):
        # x = requested points
        # xd = x data
        # yd = y data
        # p = parameters
        # func = function name
        alpha = 1.0 - conf          # significance
        N = xd.size                 # data sample size
        var_n = len(p)              # number of parameters
        # Quantile of Student's t distribution for p=(1-alpha/2)
        q = stats.t.ppf(1.0 - alpha / 2.0, N - var_n)
        # Stdev of an individual measurement
        se = np.sqrt(1. / (N - var_n) * np.sum((yd - func(xd, *p)) ** 2))
        # Auxiliary definitions
        sx = (x - xd.mean()) ** 2
        sxd = np.sum((xd - xd.mean()) ** 2)
        # Predicted values (best-fit model)
        yp = func(x, *p)
        # Prediction band
        dy = q * se * np.sqrt(1.0+ (1.0/N) + (sx/sxd))
        # Upper & lower prediction bands.
        lpb, upb = yp - dy, yp + dy
        return lpb, upb

    lpb, upb = predband(px, x, y, popt, f, conf=0.95)
    #
    # return nom, px, std, lpb, upb, r2, popt[0],popt[1],popt[2],popt[3]
    return popt[0],popt[1],popt[2],popt[3], r2, lpb, upb




fit_result = fit_function(y, [1, 2, 3, 4], x=x)



import matplotlib.pyplot as plt
plt.plot(x, y, 'o')
plt.plot(x, fit_result[0] * np.exp(fit_result[1] * (np.array(x) - fit_result[2])) + fit_result[3])
plt.plot(np.arange(1, len(fit_result[5])+1), fit_result[5])
plt.plot(np.arange(1, len(fit_result[6])+1), fit_result[6])
plt.ylabel('some numbers')
plt.show()



