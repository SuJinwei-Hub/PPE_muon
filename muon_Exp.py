import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import datetime
from scipy.integrate import quad


start = datetime.datetime.now()
usmax = 20 * int('44E', 16)


def mean_of_next(x):  # 求相邻两数的平均值
    y = []
    for i in range(0, len(x) - 1):
        y.append((x[i] + x[i + 1]) / 2)
    return np.array(y)


def func_exp(x, a, t, c):
    return a * np.exp(-x / t) + c


def func_guass(x, a, mu, sigma, y0):
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + y0


def func_cos2(x, a, b):
    return a * (np.cos(x)) ** 2 + b


class Lifetime:
    def __init__(self, txtname, i):
        self.txtname = txtname
        self.num = i

    def data(self):  # 获取数据
        f = open(self.txtname)
        item = f.readline()
        lt = []
        while item:
            if item.startswith('#'):
                lti = float(int(item[1:4], 16) * 20)
                if lti < usmax:
                    lt.append(lti / 1000)
            item = f.readline()
        return lt

    def hist_fit(self, lt, path, judge, para):
        plt.rc('font', family='simhei', size=15)
        bin = np.linspace(0, np.max(lt), 30)
        n, bins, patches = plt.hist(lt, bins=bin, edgecolor='black', alpha=0.8)
        if judge:
            x = Lifetime.bary_center(self, para, bins)
        else:
            x = mean_of_next(bins)
        y = np.array(n)
        popt, pcov = curve_fit(func_exp, x, y)
        yval = func_exp(x, popt[0], popt[1], popt[2])
        perr = np.sqrt(np.diag(pcov))
        r2 = round(np.corrcoef(y, yval)[0, 1] ** 2, 4)
        plt.plot(x, yval, label='$\\tau$ = %5.3f' %popt[1] + '$\pm$' + '%5.3f' %perr[1] + '\n' + '$R^2$ = %5.4f' % r2)
        plt.xlabel('寿命区间/$\mu$s')
        plt.ylabel('频数')
        plt.legend()
        plt.savefig(path + '\\' + self.num + "_decay.png")
        plt.close()
        return popt, perr[1]

    def bary_center(self, para, bins):      # 积分计算重心
        x = []
        for i in range(0, len(bins)-1):
            v = quad(func_exp, bins[i], bins[i+1], args=(para[0], para[1], para[2]))
            yi = v[0]/(bins[i+1] - bins[i])
            xi = para[1] * np.log(para[0]/(yi - para[2]))
            x.append(xi)
        return np.array(x)


class Flux:
    def __init__(self, txtname, i):
        self.txtname = txtname
        self.num = i

    def data(self):
        f = open(self.txtname)
        item = f.readline()
        flu = []
        while item:
            # item0代表上一列内容，item代表此列内容，这样做可以避免占用较大内存存储数组，但有可能第一个有效数据会被舍弃
            item0 = item
            item = f.readline()
            if item.startswith('&GPGGA') and item0.startswith('#'):
                fi = float(int(item0[1:4], 16))
                flu.append(fi)
        return flu

    def filte(self, x):  # 删去不良数据
        mu = np.mean(x)
        sig = np.std(x)
        de = []
        for i in range(0, len(x)):
            if np.abs(x[i] - mu) > 3 * sig:
                de.append(i)
        x = np.delete(x, de, axis=0)
        return x

    def hist_fit(self, flu, path):
        plt.rc('font', family='simhei', size=15)  # 显示汉字
        plt.rc('axes', unicode_minus=False)  # 显示负号
        n, bins, patches = plt.hist(flu, bins=int((np.max(flu) - np.min(flu)) / 2), edgecolor='black', alpha=0.8)
        x = mean_of_next(bins)
        y = np.array(n)
        mu_max = np.mean(x) * 1.8  # 用于限定拟合参数范围，防止不收敛
        sig_max = np.std(x) * 3
        param_bounds = ([0, 0, 0, 0], [10000, mu_max, sig_max, 1000])  # 指定参数的范围
        popt, pcov = curve_fit(func_guass, x, y, bounds=param_bounds)
        r2 = round(np.corrcoef(y, func_guass(x, popt[0], popt[1], popt[2], popt[3]))[0, 1] ** 2, 4)
        x = np.linspace(np.min(x), np.max(x), 50)
        yval = func_guass(x, popt[0], popt[1], popt[2], popt[3])
        plt.plot(x, yval, 'r--', label='$\Phi$ = %5.3f' %popt[1] + '$\pm$' + '%5.3f' %popt[2] + '\n' + '$R^2$ = %5.4f' % r2)
        plt.xlabel('通量区间')
        plt.ylabel('频数')
        plt.legend(loc=1)
        plt.savefig(path + '\\' + self.num + "_flux.png")
        plt.close()
        return [popt[1], popt[2]]


def cos2_fit(x, y, path):
    popt, pcov = curve_fit(func_cos2, x, y[:, 0])
    plt.errorbar(x, y[:, 0], yerr=y[:, 1], fmt="bo:", capsize=8)
    x = np.linspace(np.min(x), np.max(x), 30)
    yval = func_cos2(x, popt[0], popt[1])
    plt.plot(x, yval)
    plt.ylim([0, 20])
    plt.xlabel(path + '(rad)')
    plt.ylabel('通量')
    plt.savefig('fig_flux' + '\\' + path + "_fit.png")
    plt.close()


def decay(judge):
    fnum = 1
    pathIn = 'data_decay'
    pathOut = 'fig_decay'
    time = []
    sigma = []
    popt = 0
    judge2 = 0
    for i in range(1, fnum + 1):
        txtname = pathIn + '\\' + "muondecay_" + str(i) + ".txt"
        mu_lifetime = Lifetime(txtname, str(i))
        lt = mu_lifetime.data()
        popt, sig = mu_lifetime.hist_fit(lt, pathOut, judge2, popt)
        if judge:
            judge2 = 1
            for j in range(2):
                popt, sig = mu_lifetime.hist_fit(lt, pathOut, judge2, popt)
        time.append(popt[1])
        sigma.append(sig)
    np.savetxt('muon_lifetime.txt', time)
    return np.mean(time), np.mean(sigma)


def flux(path, x, fnum, name):
    pathIn = 'data_flux' + '\\' + path
    pathOut = 'fig_flux' + '\\' + path
    flu = []
    for i in range(1, fnum + 1):
        txtname = pathIn + '\\' + str(i) + ".txt"
        mu_flux = Flux(txtname, str(i))
        flui = mu_flux.data()
        flui = mu_flux.filte(flui)
        flu.append(mu_flux.hist_fit(flui, pathOut))
    flu = np.array(flu)
    np.savetxt(pathOut + '.txt', flu)
    plt.errorbar(x, flu[:, 0], yerr=flu[:, 1], fmt="bo:", capsize=8)
    plt.ylim([0, 20])
    plt.xlabel(path + name)
    plt.ylabel('通量')
    plt.savefig(pathOut + ".png")
    plt.close()
    if path.startswith('Direction'):
        cos2_fit(x, flu, path)


def main():
    while True:
        sw = int(input("*Select what you want to calculate {Exit(0); Lifetime(1); Flux(2)}:"))
        if sw == 0:
            print("EXIT!")
            break
        elif sw == 1:
            judge = int(input("*Iterative calculation of integral {Yes(1); No(0)}:"))
            t, s = decay(judge)
            print("The lifetime of muon is " + str(t) + '±' + str(s) + 'us')
            print("Completed!")
        elif sw == 2:
            sw2 = int(input("*Select variable {Exit(0); HV(1); Threshold(2); Direction(3)}:"))
            if sw2 == 0:
                print("EXIT!")
                break
            elif sw2 == 1:
                path = 'HV'
                x = np.arange(850, 1150 + 1, 50)
                fnum = len(x)
                name = '(V)'
            elif sw2 == 2:
                path = 'Threshold'
                x = np.arange(100, 280 + 1, 30)
                fnum = len(x)
                name = '(mV)'
            elif sw2 == 3:
                path = 'Direction'
                sw3 = int(input("*Select Direction {Exit(0); North(1); West(2)}"))
                if sw3 == 0:
                    print("EXIT!")
                    break
                elif sw3 == 1:
                    path = path + '\\' + 'North'
                elif sw3 == 2:
                    path = path + '\\' + 'West'
                x = np.array([0, 15, 30])
                x = x * np.pi / 180
                fnum = len(x)
                name = '(rad)'
            else:
                print("ERROR!")
                continue
            flux(path, x, fnum, name)
            print("Completed!")
        else:
            print("ERROR!")


main()
end = datetime.datetime.now()
print(end - start)
