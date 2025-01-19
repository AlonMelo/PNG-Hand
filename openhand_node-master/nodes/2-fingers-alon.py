import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

'''
Based on "A highly-underactuated robotic hand with force and joint angle sensors" by Wang et al. IROS, 2011
'''


class finger():
    def __init__(self, n=2):
        L = 1.
        self.n = n
        self.l = L / self.n

        self.init_angle = 26.565  # deg
        # was 20 by avishay

        self.K = 2.232  # Joint stiffness
        # was 0.6 by avishay

        self.t = 0.3 * self.l  # contact pad thickness
        # was 0.2 by avishay

        w = self.l * 0.5  # contact pad width
        # was 0.5 by avishay

        self.pad_pts = np.array([[0, 0], [w, 0], [w, self.t], [0, self.t]])
        self.tendon_dis = 0.9
        # was 0.9 by avishay
        self.f = 0.225


    def FK(self, q, contacts=False):
        q[0] += np.deg2rad(self.init_angle)

        x = 0
        y = 0
        X = [np.array([x, y])]
        X_contact = []
        l = self.l
        lc = np.sqrt(self.t ** 2 + (l / 2) ** 2)
        ac = np.arctan(self.t / (l / 2))
        for i in range(self.n):
            X_contact.append([x + lc * np.cos(np.sum(q[:i + 1]) + ac), y + lc * np.sin(np.sum(q[:i + 1]) + ac)])
            x += l * np.cos(np.sum(q[:i + 1]))
            y += l * np.sin(np.sum(q[:i + 1]))

            X.append(np.array([x, y]))

        self.X = np.array(X)
        self.X_contact = np.array(X_contact)
        if not contacts:
            return self.X
        else:
            return self.X, self.X_contact

    def R(self, theta):
        return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    def J_L(self, q, contact=False):
        X, X_contacts = self.FK(q, contacts=True)
        # X = np.concatenate((X, np.zeros((self.n+1, 1))), axis=1)

        if contact:
            p = X_contacts[-1, :]
        else:
            p = X[-1, :]

        J = []
        for i in range(self.n):
            dx = p - X[i, :]
            # j = np.cross(np.array([0,0,1]), dx)
            j = np.array([-dx[1], -dx[0], 0])  # The cross product
            J.append(j)
        self.J = np.array(J)

        return self.J

    def solve_finger_all(self, F0, f_ext=np.array([0, 0, 0])):
        # Find the conf. of the finger given external and tendon forces. Equations are solved all together

        def func(q, F0, f_ext):
            if np.any(f_ext != 0):
                JT = self.J_L(np.copy(q))
                tau = -JT.dot(f_ext)
            else:
                tau = np.zeros((self.n,))

            f = np.abs(F0 * (self.t * self.tendon_dis) + tau[-1] - self.K * q[-1])
            for i in range(self.n - 2, -1, -1):
                # f += np.abs(2*self.K/(F0*self.l)*(q[i]-q[i+1] - tau[i]) - np.sin(q[i]/2) - np.sin(q[i+1]/2))
                f += np.abs(
                    F0 * self.l / 2 * (np.sin(q[i] / 2) + np.sin(q[i + 1] / 2)) - tau[i] - self.K * (q[i] - q[i + 1]))
            return f

        while 1:
            q = np.random.random((self.n,)) * np.pi / 2
            res = minimize(func, q, method='BFGS', args=(F0, f_ext), options={'disp': False})
            if res.fun < 1e-5:
                break

        q = res.x
        print(q, res.fun)

        self.FK(q)
        self.q = q

        return q

    def solve_finger(self, F0):
        # Find the conf. of the finger given tendon force. Equations are solved sequenctially

        def func(x, x_prev, F0):
            return np.abs(2*self.K / ((F0*((1-self.f)**2)*(self.t * self.tendon_dis)+F0*(1-self.f)*self.f*self.t * self.tendon_dis) * (x - x_prev)) ) ## alon

        ############# aaaaaaaaaallllllllllllloooooooooooooonnnnnnnn
        def func2(x, x_prev, F0):
            return np.abs(40 * self.K / ((F0*((1-self.f)**2)*(self.t * self.tendon_dis)+F0*(1-self.f)*self.f*self.t * self.tendon_dis+F0*self.f*self.t * self.tendon_dis) * (x - x_prev))) ## alon

        #####################################################
        q = np.zeros((self.n,))
        # by avishay
        # q[-1] = F0 * (self.t * self.tendon_dis)/ self.K
        # print(q[-1])
        q[-1] = F0*((1-self.f)**2)*(self.t * self.tendon_dis)/self.K
        q[-2] = (F0*((1-self.f)**2)*(self.t * self.tendon_dis)+F0*(1-self.f)*self.f*self.t * self.tendon_dis)/self.K
        q[-3] = (F0*((1-self.f)**2)*(self.t * self.tendon_dis)+F0*(1-self.f)*self.f*self.t * self.tendon_dis+F0*self.f*self.t * self.tendon_dis)/(1.9*self.K)
        # for i in range(self.n - 2, -1, -1):
        #     # x0 = q[i+1]
        #     # res = minimize(func, x0, method='BFGS', args=(q[i+1], F0), options={'disp': False})
        #     # # print(i, x0, res.x)
        #     # q[i] = res.x
        #     ##3333##########aaaaaaaaaaalllllllllllloooooooooooonnnnnnnnnn
        #     x0 = q[i + 1]
        #     if i == 0:
        #         res = minimize(func2, x0, method='BFGS', args=(q[i + 1], F0), options={'disp': False})
        #     else:
        #         res = minimize(func, x0, method='BFGS', args=(q[i + 1], F0), options={'disp': False})
        #     # print(i, x0, res.x)
        #     q[i] = res.x

        print(q)

        self.FK(q)
        self.q = q

        return q

    def fit_finger(self):
        circ = (-0.315, 0.35, 0.33)

        # was (0,0.3,0.2) by avishay

        def func_q(q, circ):
            X, X_contact = self.FK(np.copy(q), contacts=True)
            f = 0
            for xc in X_contact:
                f += np.abs(np.linalg.norm(circ[:2] - xc) - circ[2])

            k = 0.02
            for i in range(1, X.shape[0]):
                xc = X_contact[i - 1]
                v1 = circ[:2] - xc
                v1 /= np.linalg.norm(v1)
                v2 = X[i] - X[i - 1]
                v2 /= np.linalg.norm(v2)
                angle = v1.dot(v2)
                f += k * np.abs(angle - 0.)
            f += 0.01 * np.sum(q)
            return f

        def func_f(x, q):
            # x = [F0, f_ext] (f_ext is the magnitude of the normal force at the last link)
            F0 = x  # [0]
            f_ext = 0  # x[1]

            _, X_contact = self.FK(np.copy(q), contacts=True)
            v = X_contact[-1] - circ[:2]
            v /= np.linalg.norm(v)
            f_ext *= v
            f_ext = np.append(f_ext, 0)

            JT = self.J_L(np.copy(q), contact=True)
            tau = -JT.dot(f_ext)

            C = np.abs(F0 * (self.t * self.tendon_dis) + 0 * tau[-1] - self.K * q[-1])
            for i in range(self.n - 2, -1, -1):
                # C += np.abs(2*self.K/(F0*self.l/2)*(q[i]-q[i+1] - tau[i]) - np.sin(q[i]/2) - np.sin(q[i+1]/2))
                C += np.abs(F0 * self.l / 2 * (np.sin(q[i] / 2) + np.sin(q[i + 1] / 2)) + 0 * tau[i] - self.K * (
                            q[i] - q[i + 1]))
            return C

        # Get pose of finger
        R = []
        for _ in range(10):
            q = np.random.random((self.n,)) * np.pi / 2
            res = minimize(func_q, q, method='BFGS', args=(circ,), options={'disp': False})
            # print("rrrreeeeessss ===" + str(res))
            R.append(res)

        F = [res.fun for res in R]
        q = R[np.argmin(F)].x
        # print("qqqqqqqqqqqqqqqqqqqqqq ===" + str(res))
        # Get forces on the finger at the given pose - we neglect friction
        R = []
        print('----')
        for _ in range(10):
            x = np.random.random((1,)) * 100
            res = minimize(func_f, x, method='BFGS', args=(np.copy(q),), options={'disp': False})
            R.append(res)
            # print(res.fun, res.x)
        print()

        F = [res.fun for res in R]
        f = R[np.argmin(F)].x
        print(f, R[np.argmin(F)].fun)  # , f[1]/f[0])
        # print('qqqqqqqqqqqqqq')
        print(q)

        self.plot_finger(q, circ=circ)

    def simulate_free_closing(self):

        F0 = np.linspace(0, 24, 100)
        # was (0, 10, 1000)
        fig, ax = plt.subplots()

        for f0 in F0:
            q = self.solve_finger(f0)
            X, X_contacts = self.FK(np.copy(q), contacts=True)
            if X[3, 0] <= -0.315:
                break
            else:
                self.plot_finger(q, Ax=ax)
                plt.draw()
                plt.pause(0.0001)
            # self.plot_finger(q, Ax=ax)
            # plt.draw()
            # plt.pause(0.0001)
        plt.show()

    def plot_finger(self, q, Ax=None, circ=None):
        X, X_contacts = self.FK(np.copy(q), contacts=True)
        print('-------------------------')
        print('Robot pose: ', X[-1, :])
        print('Robot angles: ', q)
        print('-------------------------')
        if Ax is None:
            fig, ax = plt.subplots()
        else:
            plt.clf()
            ax = Ax

        if circ is not None:
            p = plt.Circle((circ[0], circ[1]), circ[2], fc='yellow', ec='black')
            ax.add_artist(p)
        print(X[:, 0])
        plt.plot(X[:, 0], X[:, 1], 'k')
        plt.plot(X[:, 0], X[:, 1], 'or')
        plt.plot(X_contacts[:, 0], X_contacts[:, 1], 'ob')
        plt.plot(-0.63-1*X[:, 0], X[:, 1], 'k')
        plt.plot(-0.63-1*X[:, 0], X[:, 1], 'or')
        plt.plot(-0.63-1*X_contacts[:, 0], X_contacts[:, 1], 'ob')

        # Plot pads
        q[0] += np.deg2rad(self.init_angle)
        x = y = 0
        T = [np.array([0, 0.08])]
        for i in range(self.n):
            l = self.l  # if i < self.n-1 else self.L1

            pts = self.R(np.sum(q[:i + 1])).dot(self.pad_pts.T).T
            pts += np.array([x + l / 4 * np.cos(np.sum(q[:i + 1])), y + l / 4 * np.sin(np.sum(q[:i + 1]))])
            T.append(pts[0] + (pts[3] - pts[0]) * self.tendon_dis)
            T.append(pts[1] + (pts[2] - pts[1]) * self.tendon_dis)

            x += l * np.cos(np.sum(q[:i + 1]))
            y += l * np.sin(np.sum(q[:i + 1]))
            p = plt.Polygon(pts, fc='black')
            print(len(pts))
            print(pts)
            print(pts[0,0])
            pts2 = pts
            for num in range(4):
                print(num)
                pts2[num,0] = -0.63-1*pts[num,0]
            print(pts2)
            pp = plt.Polygon(pts2, fc='black')
            ax.add_artist(p)
            ax.add_artist(pp)

        # plot tendons
        T = np.array(T)
        plt.plot(T[:, 0], T[:, 1], 'c', linewidth=1)
        plt.plot(-0.63-1*T[:, 0], T[:, 1], 'c', linewidth=1)

        plt.axis('equal')

        if Ax is None:
            plt.show()


if __name__ == "__main__":
    # F = finger(4)
    # F.FK(np.array([0.,0.,0., 0.]))
    F = finger(3)
    F.FK(np.array([0., 0., 0.]))
    # F.FK(np.array([0.,0.,0.]), contacts=True)
    # f_ext = np.array([0.,0.,0])
    # q = F.solve_finger_all(3., f_ext=f_ext)

    # F.plot_finger(q)
    F.fit_finger()

    F.simulate_free_closing()
