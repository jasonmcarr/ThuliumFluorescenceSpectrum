import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from astropy.modeling import models, fitting
from scipy import optimize

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Wavelength Calibration
hor = np.genfromtxt(r"Background no Sample 10 sec exposure (900 SPF and 750LPF in Output, 700SpF Input)_USBC44471_14-36-33-738.txt", delimiter="", usecols=[0], skip_header=13)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Amplitudes Of Fluorescence
back = np.genfromtxt(r"Background no Sample 10 sec exposure (900 SPF and 750LPF in Output, 700SpF Input)_USBC44471_14-36-33-738.txt", delimiter="", usecols=[1], skip_header=13)
back2 = np.genfromtxt(r"Background no Sample 10 sec exposure (900 SPF and 750LPF in Output, 700SpF Input)_USBC44471_14-36-53-739.txt", delimiter="", usecols=[1], skip_header=13)
mgo = np.genfromtxt(r"MgO Tm LN_USBC44471_15-14-04-711.txt", delimiter="", usecols=[1], skip_header=13)
mgo2 = np.genfromtxt(r"MgO Tm LN_USBC44471_15-14-34-706.txt", delimiter="", usecols=[1], skip_header=13)
mgo3 = np.genfromtxt(r"MgO Tm LN_USBC44471_15-14-54-705.txt", delimiter="", usecols=[1], skip_header=13)
quartz = np.genfromtxt(r"Quartz 10 sec exposure (900 SPF and 750LPF in Output, 700SpF Input)_USBC44471_14-41-51-858.txt", delimiter="", usecols=[1], skip_header=13)
quartz2 = np.genfromtxt(r"Quartz 10 sec exposure (900 SPF and 750LPF in Output, 700SpF Input)_USBC44471_14-42-11-855.txt", delimiter="", usecols=[1], skip_header=13)
quartz3 = np.genfromtxt(r"Quartz 10 sec exposure (900 SPF and 750LPF in Output, 700SpF Input)_USBC44471_14-42-41-851.txt", delimiter="", usecols=[1], skip_header=13)
quartz4 = np.genfromtxt(r"Quartz 10 sec exposure (900 SPF and 750LPF in Output, 700SpF Input)_USBC44471_14-42-51-850.txt", delimiter="", usecols=[1], skip_header=13)
lnsamp1 = np.genfromtxt(r"TmLN Sample 1_USBC44471_15-03-14-527.txt", delimiter="", usecols=[1], skip_header=13)
lnsamp1_2 = np.genfromtxt(r"TmLN Sample 1_USBC44471_15-04-04-493.txt", delimiter="", usecols=[1], skip_header=13)
lnsamp1_3 = np.genfromtxt(r"TmLN Sample 1_USBC44471_15-04-24-491.txt", delimiter="", usecols=[1], skip_header=13)
lnsamp2 = np.genfromtxt(r"TmLN Sample 2_USBC44471_15-07-31-109.txt", delimiter="", usecols=[1], skip_header=13)
lnsamp2_2 = np.genfromtxt(r"TmLN Sample 2_USBC44471_15-07-51-106.txt", delimiter="", usecols=[1], skip_header=13)
lnsamp2_3 = np.genfromtxt(r"TmLN Sample 2_USBC44471_15-08-21-102.txt", delimiter="", usecols=[1], skip_header=13)
lnsamp3 = np.genfromtxt(r"TmLN Sample 3_USBC44471_15-10-42-375.txt", delimiter="", usecols=[1], skip_header=13)
lnsamp3_2 = np.genfromtxt(r"TmLN Sample 3_USBC44471_15-11-12-370.txt", delimiter="", usecols=[1], skip_header=13)
lnsamp3_3 = np.genfromtxt(r"TmLN Sample 3_USBC44471_15-11-32-368.txt", delimiter="", usecols=[1], skip_header=13)
lnundoped = np.genfromtxt(r"undoped LN 10 sec exposure (900 SPF and 750LPF in Output, 700SpF Input)_USBC44471_14-38-38-882.txt", delimiter="", usecols=[1], skip_header=13)
lnundoped2 = np.genfromtxt(r"undoped LN 10 sec exposure (900 SPF and 750LPF in Output, 700SpF Input)_USBC44471_14-39-08-878.txt", delimiter="", usecols=[1], skip_header=13)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Median Filter Function
def medfilt (x, k):
    'Apply a length-k median filter to a 1D array x. Boundaries are extended by repeating endpoints.'
    assert k % 2 == 1, 'Median filter length must be odd.'
    assert x.ndim == 1, 'Input must be one-dimensional.'
    k2 = (k - 1) // 2
    y = np.zeros ((len (x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.median (y, axis=1)

#Functions For FitB
def stdv (x, a, b):
    return np.std(x[a:b], ddof=True)
def mean (x, a, b):
    return np.mean(x[a:b])
def amp (x):
    return np.max(x)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Useful Variables
mgoavgmfilt = medfilt((mgo+mgo2+mgo3)/3 - (lnundoped+lnundoped2)/2, 5)
quartzavgmfilt = medfilt((quartz+quartz2+quartz3+quartz4)/4 - (back+back2)/2, 5)
ln1avgmfilt = medfilt((lnsamp1+lnsamp1_2+lnsamp1_3)/3 - (lnundoped+lnundoped2)/2, 5)
ln2avgmfilt = medfilt((lnsamp2+lnsamp2_2+lnsamp2_3)/3 - (lnundoped+lnundoped2)/2, 5)
lnunavgmfilt = medfilt((lnundoped+lnundoped2)/2 - (back+back2)/2, 5)

domain = np.linspace(np.min(hor), np.max(hor), 3648)

stdmgo = stdv(mgoavgmfilt, 2000, 2900)
meanmgo = mean(mgoavgmfilt, 2000, 2900)
ampmgo = amp(mgoavgmfilt)
dispmgo = np.min(mgoavgmfilt)

stdln1 = stdv(ln1avgmfilt, 2000, 2900)
meanln1 = mean(ln1avgmfilt, 2000, 2900)
ampln1 = amp(ln1avgmfilt)
displn1 = np.min(ln1avgmfilt)

stdln2 = stdv(ln2avgmfilt, 2000, 2900)
meanln2 = mean(ln2avgmfilt, 2000, 2900)
ampln2 = amp(ln2avgmfilt)
displn2 = np.min(ln2avgmfilt)

stdlnu = stdv(lnunavgmfilt, 2000, 2900)
meanlnu = mean(lnunavgmfilt, 2000, 2900)
amplnu = amp(lnunavgmfilt)
displnu = np.min(lnunavgmfilt)

stdq = stdv(quartzavgmfilt, 2000, 2900)
meanq = mean(quartzavgmfilt, 2000, 2900)
ampq = amp(quartzavgmfilt)
dispq = np.min(quartzavgmfilt)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Fit Parameters
def gaussian(x, height, center, width, offset):
    return height*np.exp(-(x - center)**2/(2*width**2)) + offset

def three_gaussians(x, h1, c1, w1, h2, c2, w2, h3, c3, w3, offset):
    return (gaussian(x, h1, c1, w1, offset=0) +
        gaussian(x, h2, c2, w2, offset=0) +
        gaussian(x, h3, c3, w3, offset=0) + offset)

errfunc3 = lambda p, x, y: (three_gaussians(x, *p) - y)**2

guess3 = [14900, 795, 30, 4500, 815, 15, 2000, 850, 50, 0]
test = three_gaussians(hor, 14900, 795, 30, 4500, 815, 15, 2000, 850, 50, 0)


g_init = models.Gaussian1D(amplitude=1, mean=700, stddev=5)
fit_g = fitting.LevMarLSQFitter()
g = fit_g(g_init, domain, mgoavgmfilt)


#x = np.linspace(0, 100, 2000)
m = models.Gaussian1D(amplitude=1, mean=795, stddev=0.1)
#m = models.Gaussian1D(amplitude=ampmgo, mean=meanmgo, stddev=stdmgo)   #depending on the data you need to give some initial valuesm = models.Gaussian1D(amplitude=ampmgo, mean=meanmgo, stddev=stdmgo)   #depending on the data you need to give some initial values
fitter = fitting.LevMarLSQFitter()
fitted_model = fitter(m, hor, mgoavgmfilt)


print(g)
print(len(mgo))
print(np.min(hor))
print(np.max(hor))

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Plotting
style.use('seaborn-white')

#BG Tests

plt.figure(10)
plt.title("Tm:MgO:LN Raw Data")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Fluorescence Intensity (arb)")
plt.plot(hor, mgo, color="#65759B", label='MgO Raw Data')
plt.plot(hor, back, color='red', label='Background')
plt.legend(loc='best')



plt.figure(1)
plt.title("Tm:MgO:LN 20nm Film")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Fluorescence Intensity (arb)")
plt.text(340, 13200, 'Pump: 670nm\nOcean Optics\nRoom Temperature')
#plt.ylim(0, 17000)
#plt.plot(hor, 1.65*fitted_model(hor), label='Gaussian', color='r')
#plt.xlim(700,950)
#plt.plot(hor, three_gaussians(hor[:,0], *optim3), label='3-Gaussian', color='b')
#plt.plot(domain, g(domain), label='Gaussian', color='r')
plt.plot(hor, mgoavgmfilt-lnunavgmfilt-quartz, color="#65759B")
#plt.fill_between(hor, 0, mgoavgmfilt, color="#65759B", alpha=0.25)
#plt.savefig('TmMgOLn20nmFilm', dpi=2000)
plt.show()

plt.figure(2)
plt.title("Quartz")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Fluorescence Intensity (arb)")
plt.text(425, 16250, 'Pump: 670nm\nOcean Optics\nRoom Temperature')
plt.ylim(0, 25000)
plt.plot(hor, quartzavgmfilt, color="#65759B")
plt.fill_between(hor, 0, quartzavgmfilt, color="#65759B", alpha=0.25)
#plt.savefig('Quartz', dpi=2000)
plt.show()

plt.figure(3)
plt.title("Tm:LN 20nm Film")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Fluorescence Intensity (arb)")
plt.text(425, 16250, 'Pump: 670nm\nOcean Optics\nRoom Temperature')
plt.ylim(0, 25000)
plt.plot(hor, ln1avgmfilt, color="#65759B")
plt.fill_between(hor, 0, ln1avgmfilt, color="#65759B", alpha=0.25)
#plt.savefig('TmLn20nmFilm', dpi=2000)
plt.show()

plt.figure(4)
plt.title("Tm:LN 20nm Film APE")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Fluorescence Intensity (arb)")
plt.text(425, 16250, 'Pump: 670nm\nOcean Optics\nRoom Temperature')
plt.ylim(0, 25000)
plt.plot(hor, ln2avgmfilt, color="#65759B")
plt.fill_between(hor, 0, ln2avgmfilt, color="#65759B", alpha=0.25)
#plt.savefig('TmLnAPE20nmFilm', dpi=2000)
plt.show()

plt.figure(6)
plt.title("Undoped LN")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Fluorescence Intensity (arb)")
plt.text(425, 16250, 'Pump: 670nm\nOcean Optics\nRoom Temperature')
plt.ylim(0, 25000)
plt.plot(hor, lnunavgmfilt, color="#65759B")
plt.fill_between(hor, 0, lnunavgmfilt, color="#65759B", alpha=0.25)
#plt.savefig('UndopedLN', dpi=2000)
plt.show()

