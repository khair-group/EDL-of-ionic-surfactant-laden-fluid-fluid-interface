import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from matplotlib import rcParams, ticker

rcParams['text.usetex'] = True
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams['text.latex.preamble'] = r"""\usepackage{amsfonts, amsmath, amssymb, amsthm, bm, cancel, upgreek}"""
rcParams['font.size'] = 16
rcParams['legend.fontsize'] = 12
rcParams['legend.frameon'] = False
rcParams['savefig.bbox'] = 'tight'
rcParams['figure.subplot.wspace'] = 0.4  # Horizontal space between subplots
rcParams['figure.subplot.hspace'] = 0.4  # Vertical space between subplots

figname = 'fig{fignum:1d}.png'
fdim = (8,3.5) # width, height
fdim2 = (4,4)
dpi = 1200
layout = 'constrained'
#h_space = 0.05
options = {'figsize': fdim, 'dpi': dpi}#, 'layout': layout}
option2 = {'figsize': fdim2, 'dpi': dpi}#, 'layout': layout}

def psi_inner(x, eps=1e-2, delta=0.0, beta=1):
    arg = x * eps**(-4/3) * (1 + delta)**(1/2)
    param = (2 * beta * (1+delta)**(1/2))**(1/3)
    out = -2 * np.log( param + arg/2) - (2/3) * np.log(eps)
    return out

def psi_debye(x, eps=1e-2, delta=0.0, beta=1):
    arg = (x/eps) * (1+delta)**(1/2)
    top = 1 + np.exp(-arg)
    bot = 1 - np.exp(-arg)
    out = 2*np.log(top/bot) 
    return out

def composite(x, eps=1e-2, delta=0.0, beta=1):
    out =  psi_debye(x, eps, delta, beta) + psi_inner(x, eps, delta, beta) - 2 * np.log(2/(x * (1+delta)**(1/2) / eps)) 
    return out

def tension(c, K=1e-6, T=298, G_infty=2.5e-6, g_0=52e-3, dim=False, as_surf_pressure=False):
    k_B = 1.380649e-23 # J/K
    N = 6.02e23
    # c must be in m M [mol/m^3]
    # G_infty in mol/m^2, default from Bonfillon
    # for water/dodecane bare interface, g_0 = 52 dyn/cm = 52e-3 N/m, default from Bonfillon
    # K = 1e-6 comes from Chang et al. for an air/water interface, just using this as a back of the envelope
    epsi = get_upeps(c, g_inf=G_infty)
    prefactor = G_infty * k_B * T * N / g_0
    out = 1 - epsi**(2/3) * prefactor * ( 3 * ( 4 * K * c )**(1/3) - 8 * epsi**(1/3) )
    if as_surf_pressure: out = 1 - out
    if dim: out *= g_0 # output in N/m
    return out

def get_upeps(c, g_inf=2.5e-6):
    k_B = 1.380649e-23 # J/K
    e = 1.602e-19 # C
    eps = 80 * 8.85418782e-12 # F/m [=] C^2 / N m^2
    N = 6.02e23
    T = 298
    etop = eps * k_B * T * c * N
    ebot = 2 * g_inf**2 * e**2 * N**2
    out = ( etop / ebot )**(1/2)
    return out

def get_num_sol(eps=1e-2, delta=0.0, beta=1.0, return_zetas=False, return_gammas=False, N_Nodes=2500):
    zetas = []
    gammas = []

    def get_gam(zeta):
        top = np.exp(-zeta)
        bot = beta + np.exp(-zeta)
        out = top/bot
        return out

    def f(qqqq, y, e=1e-2, d=0.0):
        y1, dy1 = y
        dy2 = (1 + d) * (e**-2) * np.sinh(y1)
        return np.vstack((dy1, dy2))

    def bc(ya, yb, e=1e-2):
        y_left, dy_left, right_bdy = ya[0], ya[1], yb[0]
        left_bdy = dy_left + (e**-2)/2 / (1 + beta * np.exp(y_left))
        residual = np.array([left_bdy, right_bdy])
        return residual
    
    if hasattr(eps, '__iter__'):
        lowerlim = int(round(np.log10(min(eps)**(-2)), 1))
    else:
        lowerlim = int(round(np.log10(eps**(-2)), 1))
    
    x = np.logspace(-lowerlim, 2, N_Nodes)
    x -= x[0]
    y0 = np.ones((2, x.size))
    if hasattr(eps, '__iter__'):
        for e in eps:
            print('eps = ')
            print(e)
            print('delta = ')
            print(delta)
            sol = solve_bvp(
                fun=lambda a,b: f(a, b, e=e, d=delta),
                bc=lambda a,b: bc(a, b, e=e),
                x=x,
                y=y0,
                verbose=2,
                max_nodes=1000000
            )
            x = sol.x
            y0 = sol.y
            zeta_potential = sol.sol(x)[0][0]
            zetas.append(zeta_potential)
            gamma = get_gam(zeta_potential)
            gammas.append(gamma)
            print('\n')

    elif hasattr(delta, '__iter__'):
        for d in delta:
            print('eps = ')
            print(eps)
            print('delta = ')
            print(d)
            sol = solve_bvp(
                fun=lambda a,b: f(a, b, e=eps, d=d),
                bc=lambda a,b: bc(a, b, e=eps),
                x=x,
                y=y0,
                verbose=2,
                max_nodes=1000000
            )
            x = sol.x
            y0 = sol.y
            zeta_potential = sol.sol(x)[0][0]
            zetas.append(zeta_potential)
            gamma = get_gam(zeta_potential)
            gammas.append(gamma)
            print('\n')
    else:
        print('eps = ')
        print(eps)
        print('delta = ')
        print(delta)
        sol = solve_bvp(
                fun=lambda a,b: f(a, b, e=eps, d=delta),
                bc=lambda a,b: bc(a, b, e=eps),
                x=x,
                y=y0,
                verbose=2,
                max_nodes=1000000
            )
        print('\n')
    
    if return_gammas and not return_zetas:
        out = gammas
    elif not return_gammas and return_zetas:
        out = zetas
    elif return_gammas and return_zetas:
        out = gammas, zetas
    else:
        out = sol

    return out

def inverse_geteps(epsi):
    g_inf=2.5e-6
    k_B = 1.380649e-23 # J/K
    e = 1.602e-19 # C
    eps = 80 * 8.85418782e-12 # F/m [=] C^2 / N m^2
    N = 6.02e23
    T = 298
    atop = eps * k_B * T * N
    abot = 2 * g_inf**2 * e**2 * N**2
    alpha = np.sqrt(atop/abot)
    out = (epsi/alpha)**2
    return out

def get_dtab():
    # x is DTAB conc in mol/m^3
    # y is surf pressure in mN/m
    # 0.5 mol/dm^3
    x1 = [0.01988652629310696, 0.031829017668437556, 0.06284824112125471, 0.12277848854414006, 0.23778646125439618, 0.48769806481213956, 0.9195254247037922, 1.8516275208482682]
    y1 = [11.762597602082177, 14.216409950078935, 17.815846738063744, 22.44613218415326, 27.074284251397362, 32.407731364935785, 37.56922814353373, 42.999103980884925]

    # 0.25 mol/dm^3
    x2 = [0.04082895206072309, 0.07755825381331166, 0.1541395074078651, 0.3132964817515189, 0.6125330058341673, 1.1988455516672523, 2.3605428152057915]
    y2 = [13.828135000213338, 17.713017877714723, 21.078209668472926, 25.669667619575883, 29.91125144003072, 35.46571660195417, 40.36011434910611]

    # 0.2 mol/dm^3
    x3 = [0.04097102529244394, 0.07905034428712507, 0.15678298116338346, 0.31429307235486076, 0.6192773186803633, 1.2160216532135373, 2.3882712367385306, 4.748185048019814]
    y3 = [10.552971796731663, 13.815761402909928, 17.116525152536585, 20.978794214276572, 25.255792123565303, 30.130136109570337, 35.218671331655074, 41.23607970303367]

    # 0.07 mol/dm^3
    x4 = [0.054532584693138514, 0.1069924239047236, 0.21440300753341596, 0.432524763020743, 0.8308309198458672, 1.6387880176108884, 3.2571477488906044, 6.436109722442438]
    y4 = [12.094978026197891, 15.012160259418867, 17.81115330460383, 21.859026325894952, 25.790416862226394, 31.126424030379315, 36.39117634509536, 42.39919784955412]

    x1, x2, x3, x4 = np.array(x1), np.array(x2), np.array(x3), np.array(x4)
    xs = [x1, x2, x3, x4]
    y1, y2, y3, y4 = np.array(y1), np.array(y2), np.array(y3), np.array(y4)
    ys = [y1, y2, y3, y4]

    d1 = 0.5 * 10**3
    d2 = 0.25 * 10**3
    d3 = 0.2 * 10**3
    d4 = 0.07 * 10**3
    ds = (d1, d2, d3, d4)
    deltas1 = d1 / x1
    deltas2 = d2 / x2
    deltas3 = d3 / x3
    deltas4 = d4 / x4
    deltas = [deltas1, deltas2, deltas3, deltas4]
    return xs, ys, ds, deltas


fc = 1

beta, delta = 1, 0
continuous_epsilons = np.logspace(-6, 0, 1000)
gamma_asymp_pred = (2 * continuous_epsilons)**(2/3) * beta**(-1/3)
zeta_asymp_pred = -(2/3) * np.log(2 * continuous_epsilons * beta)

epsilons = np.logspace(-6, 0, 30)
gammas, zetas = get_num_sol(epsilons, return_gammas=True, return_zetas=True)


fig, (ax1, ax2) = plt.subplots(1, 2, num=fc, **options)
ax1.loglog(continuous_epsilons, gamma_asymp_pred, '-k', label=r'asymptotic')
ax1.loglog(epsilons, gammas, 'ok', label='numerical', markerfacecolor="None", markeredgecolor='black')

ax2.loglog(continuous_epsilons, zeta_asymp_pred, '-k', label=r'asymptotic')
ax2.loglog(epsilons, zetas, 'ok', label='numerical', markerfacecolor="None", markeredgecolor='black')

ax1.set_xlabel(r'$\varepsilon$' + '\n(a)')
ax1.set_ylabel(r'$\Gamma$', rotation=0, labelpad=10)

ax1.set_xlim((min(continuous_epsilons), max(continuous_epsilons)))
ax1.set_ylim((min(gammas), max(gammas)))

ax2.set_xlabel(r'$\varepsilon$' + '\n(b)')
ax2.set_ylabel(r'$\zeta$', rotation=0, labelpad=10)
ax2.set_xlim((min(continuous_epsilons), max(continuous_epsilons)))
ax2.set_ylim((min(zetas), 10))

nticks = 9
maj_loc = ticker.LogLocator(numticks=nticks)
min_loc = ticker.LogLocator(subs='all', numticks=9)
ax1.xaxis.set_major_locator(maj_loc)
ax1.xaxis.set_minor_locator(min_loc)
ax2.xaxis.set_major_locator(maj_loc)
ax2.xaxis.set_minor_locator(min_loc)

for i, labels in enumerate(zip(ax1.get_xticklabels(), ax2.get_xticklabels())): 
    label1, label2 = labels
    if i % 2 == 0: 
        label1.set_visible(False)
        label2.set_visible(False)
ax1.legend()
ax2.legend()

plt.savefig(figname.format(fignum=2))

fc +=1 

eps=1e-2
sol = get_num_sol(eps=eps)
psi = sol.sol
cx = np.logspace(-7, 1, 1000)
x = np.logspace(-7, 1, 18)
c_p_num = np.exp(-psi(x)[0])
c_n_num = np.exp(psi(x)[0])

asymp_c_p = np.exp(-composite(cx, eps=eps))
asymp_c_n = np.exp(composite(cx, eps=eps))
a_cp_in = np.exp(-psi_inner(cx, eps=eps))
a_cp_out = np.exp(-psi_debye(cx, eps=eps))
a_cn_in =  np.exp(psi_inner(cx, eps=eps))
a_cn_out = np.exp(psi_debye(cx, eps=eps))

fig, (ax3, ax4) = plt.subplots(1, 2, num=fc, **options)

ax3.loglog(cx, asymp_c_p, '-k', label='composite')
ax3.loglog(cx, a_cp_in, ':k', label='inner')
ax3.loglog(cx, a_cp_out, '--k', label='Debye')
ax3.loglog(x, c_p_num, 'ok', label='numerical', markerfacecolor="None", markeredgecolor='black')
ax3.set_ylabel(r'$c_+$', labelpad=10, rotation=0)
ax3.set_xlabel(r'$x$' + '\n(a)')
ax3.set_xlim([min(x), max(x)])
ax3.set_ylim([0.1*min(asymp_c_p), 10*max(asymp_c_p)])

ax4.loglog(cx, asymp_c_n, '-k', label='composite')
ax4.loglog(cx, a_cn_in, ':k', label='inner')
ax4.loglog(cx, a_cn_out, '--k', label='Debye')
ax4.loglog(x, c_n_num, 'ok', label='numerical', markerfacecolor="None", markeredgecolor='black')
ax4.set_ylabel(r'$c_-$', labelpad=10, rotation=0)
ax4.set_xlabel(r'$x$' + '\n(b)')
ax4.set_xlim([min(x), max(x)])
ax4.set_ylim([0.1*min(asymp_c_n), 10*max(asymp_c_n)])

nticks = 9
maj_loc = ticker.LogLocator(numticks=nticks)
min_loc = ticker.LogLocator(subs='all', numticks=9)
ax3.xaxis.set_major_locator(maj_loc)
ax3.xaxis.set_minor_locator(min_loc)
ax4.xaxis.set_major_locator(maj_loc)
ax4.xaxis.set_minor_locator(min_loc)

for i, labels in enumerate(zip(ax3.get_xticklabels(), ax4.get_xticklabels())): 
    label1, label2 = labels
    if i % 2 == 0: 
        label1.set_visible(False)
        label2.set_visible(False)

ax3.legend()
ax4.legend()

plt.savefig(figname.format(fignum=4))
fc += 1

eps = 1e-3
beta = 1
deltas = np.logspace(-1, 7, 40)
cdelts = np.logspace(-1, 7, 1000)

gamma_asymp_pred = (2 * eps)**(2/3) * ((1 + cdelts)/beta)**(1/3)
zeta_asymp_pred = -(2/3) * np.log(2 * eps * beta * (1+cdelts)**(1/2))

gammas, zetas = get_num_sol(eps=eps, delta=deltas, return_gammas=True, return_zetas=True, N_Nodes=75000)
max_of_first_zetas = max(zetas)
min_of_first_gammas = min(gammas)
fig, (ax1, ax2) = plt.subplots(1, 2, num=fc, **options)

a1line1, = ax1.loglog(cdelts, gamma_asymp_pred, '-k', label=r'asymptotic')
a1line2, = ax1.loglog(deltas, gammas, 'ok', label=r'numerical', markerfacecolor="None", markeredgecolor='black')

a2line1, = ax2.loglog(cdelts, zeta_asymp_pred, '-k', label=r'asymptotic')
a2line2, = ax2.loglog(deltas, zetas, 'ok', label=r'numerical', markerfacecolor="None", markeredgecolor='black')

eps = 1e-2
gammas, zetas = get_num_sol(eps=eps, delta=deltas, return_gammas=True, return_zetas=True, N_Nodes=75000)
gamma_asymp_pred = (2 * eps)**(2/3) * ((1 + cdelts)/beta)**(1/3)
zeta_asymp_pred = -(2/3) * np.log(2 * eps * beta * (1+cdelts)**(1/2))


a1line3, = ax1.loglog(cdelts, gamma_asymp_pred, '-k')
a1line4, = ax1.loglog(deltas, gammas, 'ok', markerfacecolor="None", markeredgecolor='black')

a2line3, = ax2.loglog(cdelts, zeta_asymp_pred, '-k')
a2line4, = ax2.loglog(deltas, zetas, 'ok', markerfacecolor="None", markeredgecolor='black')

label1 = r'$\varepsilon=10^{-3}$'
label2 = r'$\varepsilon=10^{-2}$'
label1xy = (10**5, 0.1)
textxy1 = (10**5, 0.1)
rot1 = 0

label2xy = (3.4, 0.3)
textxy2 = (3.4, 0.3)
rot2 = 0
text = ax1.annotate(label1, label1xy, xytext=textxy1, ha='center', rotation=rot1)
text = ax1.annotate(label2, label2xy, xytext=textxy2, ha='center', rotation=rot2)
text.set_bbox(dict(facecolor='white', edgecolor='white', alpha=0.0))

label1xy = (10**5, 2)
textxy1 = (10**5, 2)
rot1 = 0

label2xy = (2.5, 0.7)
textxy2 = (2.5, 0.7)
rot2 = 0

text = ax2.annotate(label1, label1xy, xytext=textxy1, ha='center', rotation=rot1)
text = ax2.annotate(label2, label2xy, xytext=textxy2, ha='center', rotation=rot2)
text.set_bbox(dict(facecolor='white', edgecolor='white', alpha=0.0))

ax1.set_xlabel(r'$\delta$' + '\n(a)')
ax1.set_ylabel(r'$\Gamma$', rotation=0, labelpad=10)

ax1.set_xlim((min(deltas), max(deltas)))
ax1.set_ylim((1e-2, max(gammas)))

ax2.set_xlabel(r'$\delta$' + '\n(b)')
ax2.set_ylabel(r'$\zeta$', rotation=0, labelpad=10)
ax2.set_xlim((min(deltas), max(deltas)))
ax2.set_ylim((min(zetas), max_of_first_zetas))

nticks = 9
maj_loc = ticker.LogLocator(numticks=nticks)
min_loc = ticker.LogLocator(subs='all', numticks=9)
ax1.xaxis.set_major_locator(maj_loc)
ax1.xaxis.set_minor_locator(min_loc)
ax2.xaxis.set_major_locator(maj_loc)
ax2.xaxis.set_minor_locator(min_loc)

for i, labels in enumerate(zip(ax1.get_xticklabels(), ax2.get_xticklabels())): 
    label1, label2 = labels
    if i % 2 == 0: 
        label1.set_visible(False)
        label2.set_visible(False)


plt.savefig(figname.format(fignum=6))

fc +=1 


warczinsky_raw_x = [-4.523939753997704, -4.397865236410464, -4.302856308372442, -4.2234598304668465, -4.0494527013123225, -3.9980949093588407, -3.8231717435894033, -3.700880855498543, -3.6041792511068778, -3.400161965894231, -3.3061288159894855, -3.160259941984348, -3.0499571852451726, -3.0021042290357056, -2.8288139981015474, -2.702759394353763, -2.606237014517189, -2.524271651321947, -2.4004075699141714, -2.3033076887334136]
warczinsky_raw_y = [72.44856737691805, 72.21898407388554, 72.20938477201658, 72.13918987709985, 72.03399752745256, 71.9252054396044, 71.82481274089157, 71.541833321213, 71.32284924732747, 70.46791142462367, 69.51498072867427, 68.05268707730346, 66.49100065449785, 65.73665551596247, 62.527488909897464, 59.74029161515526, 57.04468765907934, 54.6700603592466, 50.789542578721544, 47.845156715875206]
w_x = np.array(warczinsky_raw_x) # warz. x data in log_10 conc. mol/dm^3
w_y = np.array(warczinsky_raw_y) # warz. y data in mN/m
w_x = 10**w_x
w_x *= 1e3
w_x_contin = np.logspace(1.1*np.log10(min(w_x)), 1.5*np.log10(max(w_x)), 500)
w_y *= 1e-3 # now in N/m
w_sp_y = 0.0725 - w_y # surface pressure
s_pres = tension(w_x_contin, g_0 = 72e-3, K=2e2, G_infty=2.5e-6, as_surf_pressure=True, dim=True)
fig, ax = plt.subplots(1, 1, squeeze=True, num=fc, **option2)
ax.loglog(w_x_contin, s_pres, '-k', label='asymptotic prediction (60)')
ax.loglog(w_x, w_sp_y, 'ok', label='Warszynski \emph{et al.} (1998)', markerfacecolor="None", markeredgecolor='black')
secax = ax.secondary_xaxis('top', functions=(get_upeps, inverse_geteps))
secax.set_xlabel(r'$\varepsilon$')
ax.set_xlabel(r'$c_\infty$ [mM]')
ax.set_ylabel(r'$\Pi$ [N/m]', labelpad=5)
ax.legend()
plt.savefig(figname.format(fignum=5))

fc += 1

xs, ys, ds, deltas = get_dtab()
d1, d2, d3, d4 = ds
x1, x2, x3, x4 = xs
y1, y2, y3, y4 = ys
deltas1, deltas2, deltas3, deltas4 = deltas

fig, (ax1, ax2) = plt.subplots(1, 2, squeeze=True, num=fc, **options)
ax1.plot(x1, y1 * 1e-3, 'sk', label=r'500 [mM]')
ax1.plot(x2, y2 * 1e-3, 'ok', label=r'250 [mM]')
ax1.plot(x3, y3 * 1e-3, 'sk', label=r'200 [mM]', markerfacecolor="None", markeredgecolor='black')
ax1.plot(x4, y4 * 1e-3, 'ok', label=r'70 [mM]', markerfacecolor="None", markeredgecolor='black')
ax1.set_ylabel(r'$\Pi$ [N/m]')
ax1.set_xlabel(r'$c_\infty$ [mM]' + '\n(a)')

ax2.plot(x1**(2/3) * (1 + deltas1)**(1/3), y1 * 1e-3, 'sk', label=r'500 [mM]')
ax2.plot(x2**(2/3) * (1 + deltas2)**(1/3), y2 * 1e-3, 'ok', label=r'250 [mM]')
ax2.plot(x3**(2/3) * (1 + deltas3)**(1/3), y3 * 1e-3, 'sk', label=r'200 [mM]', markerfacecolor="None", markeredgecolor='black')
ax2.plot(x4**(2/3) * (1 + deltas4)**(1/3), y4 * 1e-3, 'ok', label=r'70 [mM]', markerfacecolor="None", markeredgecolor='black')
ax2.set_xlabel(r'$c_\infty^{2/3} \left(1 + \delta \right)^{1/3} \,  [\text{mM}^{2/3}]$' + '\n(b)')
ax2.set_ylabel(r'$\Pi$ [N/m]')

ax1.legend()
ax2.legend()
plt.savefig(figname.format(fignum=7))


fc += 1
eps = 1e-2
sol = get_num_sol(eps=eps)
psi = sol.sol
x = np.logspace(-7, 1, 18)
cx = np.logspace(-7, 1, 5000)
fig, (ax1, ax2) = plt.subplots(1, 2, num=fc, **options)
ax1.semilogx(x, psi(x)[0], 'ok', label='numerical', markerfacecolor="None", markeredgecolor='black')
ax1.semilogx(cx, psi_inner(cx, eps), ':k', label='inner')
ax1.semilogx(cx, psi_debye(cx, eps), '--k', label='Debye')
ax1.semilogx(cx, composite(cx, eps), '-k', label='composite')
ax1.set_xlabel(r'$x$'+ '\n(a)')
ax1.set_ylabel(r'$\psi$', rotation=0, labelpad=10)
ax1.set_xlim([min(x), max(x)])
ax1.set_ylim([-0.2, 1.1*max(psi(x)[0])])

eps = 1e-4
sol = get_num_sol(eps=eps)
psi = sol.sol
ax2.semilogx(x, psi(x)[0], 'ok', label='numerical', markerfacecolor="None", markeredgecolor='black')
ax2.semilogx(cx, psi_inner(cx, eps), ':k', label='inner')
ax2.semilogx(cx, psi_debye(cx, eps), '--k', label='Debye')
ax2.semilogx(cx, composite(cx, eps), '-k', label='composite')
ax2.set_xlabel(r'$x$'+ '\n(b)')
ax2.set_ylabel(r'$\psi$', rotation=0, labelpad=10)
ax2.set_xlim([min(x), max(x)])
ax2.set_ylim([-0.1, 1.1*max(psi(x)[0])])

label1 = r'$\varepsilon=10^{-2}$'
label2 = r'$\varepsilon=10^{-4}$'
label1xy = (10**-5, 1.5)
textxy1 = (10**-5, 1.5)
rot1 = 0

label2xy = (1e-1, 2)
textxy2 = (1e-1, 2)
rot2 = 0
text1 = ax1.annotate(label1, label1xy, xytext=textxy1, ha='center', rotation=rot1)
text2 = ax2.annotate(label2, label2xy, xytext=textxy2, ha='center', rotation=rot2)
text1.set_bbox(dict(facecolor='white', edgecolor='white', alpha=0.0))
text2.set_bbox(dict(facecolor='white', edgecolor='white', alpha=0.0))

nticks = 9
maj_loc = ticker.LogLocator(numticks=nticks)
min_loc = ticker.LogLocator(subs='all', numticks=9)
ax1.xaxis.set_major_locator(maj_loc)
ax1.xaxis.set_minor_locator(min_loc)
ax2.xaxis.set_major_locator(maj_loc)
ax2.xaxis.set_minor_locator(min_loc)

for i, labels in enumerate(zip(ax1.get_xticklabels(), ax2.get_xticklabels())): 
    label1, label2 = labels
    if i % 2 == 0: 
        label1.set_visible(False)
        label2.set_visible(False)

ax1.legend()
ax2.legend()

plt.savefig(figname.format(fignum=3))