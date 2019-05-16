import numpy as np
import abc


class TwoLayerDifferenceScheme(object):
    def __init__(self, t0, te, x0, xe, xsteps, tsteps, init: callable):
        self.t = t0
        self.te = te
        self.xgrid = np.linspace(x0, xe, xsteps + 1)
        self.h = self.xgrid[1] - self.xgrid[0]
        self.tgrid = np.linspace(t0, te, tsteps + 1)
        self.tau = self.tgrid[1] - self.tgrid[0]
        self.u_current = init(self.t, self.xgrid)

    @abc.abstractmethod
    def step(self):
        raise NotImplementedError

    def solve(self):
        yield self.u_current
        while self.t < self.te:
            self.step()
            yield self.u_current


class RiemannTwoLayerDifferenceScheme(TwoLayerDifferenceScheme):
    def __init__(self, t0, te, x0, xe, xsteps, tsteps, init: callable, r_plus: callable, r_minus: callable):
        super(RiemannTwoLayerDifferenceScheme, self).__init__(t0, te, x0, xe, xsteps, tsteps, init)
        self.r_plus_current = r_plus(self.u_current)
        self.r_minus_current = r_minus(self.u_current)

    @abc.abstractmethod
    def step(self):
        raise NotImplementedError


class RiemannKIRBurgerInviscid(RiemannTwoLayerDifferenceScheme):
    def step(self):
        u = np.zeros_like(self.u_current)
        r_plus = np.zeros_like(self.r_plus_current)
        r_minus = np.zeros_like(self.r_minus_current)
        r_plus[0] = self.r_plus_current[0]
        r_plus[-1] = self.r_plus_current[-1]
        r_minus[0] = self.r_minus_current[0]
        r_minus[-1] = self.r_minus_current[-1]
        u[0] = self.u_current[0]
        u[-1] = self.u_current[-1]
        for i in range(1, len(u)-1):
            r_plus[i] = self.r_plus_current[i] - self.tau*self.u_current[i] * \
                        (self.r_plus_current[i] - self.r_plus_current[i-1]) / self.h
            r_minus[i] = self.r_minus_current[i] + self.tau * (1 / 3) * self.u_current[i] * \
                         (self.r_minus_current[i+1] - self.r_minus_current[i]) / self.h
        u = r_plus/2 + 3*r_minus/2
        self.u_current = u
        self.r_minus_current = r_minus
        self.r_plus_current = r_plus
        self.t += self.tau


class KIRBurgerInviscid(TwoLayerDifferenceScheme):
    def step(self):
        u = np.zeros_like(self.u_current)
        u[0] = self.u_current[0]
        u[-1] = self.u_current[-1]
        for i in range(1, len(u) - 1):
            u[i] = self.u_current[i]-(self.tau/(2*self.h))*(self.u_current[i]**2-self.u_current[i-1]**2)
        self.u_current = u
        self.t += self.tau


class LaxWendroffInviscid(TwoLayerDifferenceScheme):
    def step(self):
        u = np.zeros_like(self.u_current)
        u[0] = self.u_current[0]
        u[-1] = self.u_current[-1]
        for i in range(2, len(u) - 1):
            #u[i] = self.u_current[i]-(self.tau/(2*self.h))*(self.u_current[i]**2-self.u_current[i-1]**2)
            u[i] = self.u_current[i] -\
                   0.5*self.tau/self.h*(0.5*self.u_current[i+1]**2 - 0.5*self.u_current[i-1]**2) + \
                   0.5*(self.tau/self.h)**2*((0.5*(self.u_current[i] + self.u_current[i+1]))*
                                             (0.5*self.u_current[i+1]**2 - 0.5*self.u_current[i]**2) -
                                             (0.5*(self.u_current[i] + self.u_current[i-1]))*
                                             (0.5*self.u_current[i]**2 - 0.5*self.u_current[i-1]**2))
        self.u_current = u
        self.t += self.tau


def _solve_and_describe(method_class, *args, **kwargs):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import numpy as np

    method = method_class(*args)
    solution = np.array(list(method.solve()))
    ts = method.tgrid
    xs = method.xgrid

    plt.plot(xs, solution[0])
    plt.plot(xs, solution[10])
    plt.plot(xs, solution[100])
    plt.plot(xs, solution[200])
    plt.plot(xs, solution[300])
    plt.plot(xs, solution[400])
    plt.plot(xs, solution[500])
    plt.plot(xs, solution[600])
    plt.plot(xs, solution[700])
    plt.plot(xs, solution[800])
    plt.plot(xs, solution[900])
    plt.plot(xs, solution[1000])
    plt.grid()
    if kwargs.get('title'):
        plt.title(kwargs.get('title'))
    plt.show()

if __name__ == '__main__':

    #I = lambda t, x: np.exp(-2*(x-1)**2)
    #I = lambda t, x: x < 2
    R_plus = lambda u: u/2 - u**3
    R_minus = lambda u: u/2 + u**3 / 3
    # Графики будут отображены в том порядке, в котором следуют функции
    I_list = [
        lambda t, x: np.exp(-2*(x-1)**2),
        lambda t, x: np.sin(np.exp(-(x-1)**2)),
        lambda t, x: x < 2,
        lambda t, x: x < t/2,
    ]
    #method = RiemannKIRBurgerInviscid(0, 4, -1, 4, 1000, 1000, I, R_plus, R_minus)
    #method = KIRBurgerInviscid(0, 4, -1, 4, 1000, 1000, I)
    for func in I_list:
        args_solver = [0, 4, -1, 4, 1000, 1000, func]
        _solve_and_describe(RiemannKIRBurgerInviscid, *(args_solver +[R_plus, R_minus]), title='KIR с инвариантами Римана')
        _solve_and_describe(LaxWendroffInviscid, *args_solver, title='Схема Лакса-Вендроффа')
        _solve_and_describe(KIRBurgerInviscid, *args_solver, title='KIR без инвариантов Римана')

    '''

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X, Y = np.meshgrid(xs, ts)

    surf = ax.plot_surface(X, Y, solution, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    '''
