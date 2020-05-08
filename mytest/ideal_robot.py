#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib
matplotlib.use('nbagg')
import matplotlib.animation as anm
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.patches as patches

class World(object):
    def __init__(self, time_span, time_interval, debug=False):
        self.objects = []
        self.time_span= time_span
        self.time_interval= time_interval
        self.debug = debug
    
    def append(self, obj):
        self.objects.append(obj)
    
    def draw(self):
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel('X', fontsize=20)
        ax.set_ylabel('Y', fontsize=20)
        
        elems = []
        
        if self.debug:
            for i in range(1000):
                self.one_step(i, elems, ax)
        else:
            self.ani = anm.FuncAnimation(fig, self.one_step, fargs=(elems, ax), frames=int(self.time_span/self.time_interval)+1, interval=int(self.time_interval*1000), repeat=False)
            plt.show()

    def one_step(self, i, elems, ax):
        while elems:
            elems.pop().remove()
        time_str = "t = {:.2f}[s]".format(i*self.time_interval)
        elems.append(ax.text(-4.4, 4.5, time_str, fontsize=10))
        for obj in self.objects:
            obj.draw(ax, elems)
            if hasattr(obj, "one_step"):
                obj.one_step(1.0)


# In[47]:


class IdealRobot(object):
    def __init__(self, pose, agent=None, sensor=None, color='black'):
        self.pose = pose
        self.r = 0.2
        self.agent = agent
        self.color = color
        self.poses = [pose]
        self.sensor = sensor
    
    def draw(self, ax, elems):
        x, y, theta = self.pose
        xn = x + self.r*math.cos(theta)
        yn = y + self.r*math.sin(theta)
        elems += ax.plot([x, xn], [y, yn], color=self.color)
        c = patches.Circle(xy = (x,y), radius=self.r, fill=False, color=self.color)
        elems.append(ax.add_patch(c))
        
        self.poses.append(self.pose)
        elems += ax.plot([e[0] for e in self.poses], [e[1] for e in self.poses], linewidth=0.5, color='black')
        if self.sensor and len(self.poses) > 1:
            self.sensor.draw(ax, elems, self.poses[-2])
        if self.agent and hasattr(self.agent, "draw"):
            self.agent.draw(ax, elems)
    
    def one_step(self, time_interval):
        if not self.agent:
            return
        obs = self.sensor.data(self.pose) if self.sensor else None
        nu, omega = self.agent.decision(obs)
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)
    
    @classmethod
    def state_transition(cls, nu, omega, dt, pose):
        theta = pose[2]
        if math.fabs(omega) < 1e-5:
            return pose + np.array([nu*math.cos(theta), nu*math.sin(theta), omega]) * dt
        else:
            return pose + np.array([nu/omega*(math.sin(theta+omega*dt)-math.sin(theta)), nu/omega*(-math.cos(theta+omega*dt)+math.cos(theta)), omega*dt])


class Agent(object):
    def __init__(self, nu, omega):
        self.nu = nu
        self.omega = omega
    
    def decision(self, observation=None):
        return self.nu, self.omega

# In[9]:


class LandMark(object):
    def __init__(self, x, y):
        self.pos = np.array([x, y]).T
        self.id = None
    
    def draw(self, ax, elems):
        c = ax.scatter(self.pos[0], self.pos[1], s=100, marker="*", label="landmarks", color="orange")
        elems.append(c)
        elems.append(ax.text(self.pos[0], self.pos[1], "LM"+str(self.id), fontsize=10))


# In[10]:


class Map(object):
    def __init__(self):
        self.landmarks = []
    
    def append_landmark(self, lm):
        lm.id = len(self.landmarks)
        self.landmarks.append(lm)
    
    def draw(self, ax, elems):
        for l in self.landmarks:
            l.draw(ax, elems)



# In[61]:


class IdealCamera(object):
    def __init__(self, envmap, range_dist=(0.5, 6.0), range_dir=(-math.pi/3, math.pi/3)):
        self.map = envmap
        self.lastdata = []
        self.range_dist = range_dist
        self.range_dir = range_dir
    
    def data(self, cam_pose):
        observed = []
        for lm in self.map.landmarks:
            ob = self.observe(cam_pose, lm.pos)
            if self.visible(ob[0], ob[1]):
                observed.append((ob, lm.id))
        self.lastdata = observed
        return observed
    
    def visible(self, dis, phi):
        if not (self.range_dist[0] < dis < self.range_dist[1]):
            return False
        if not (self.range_dir[0] < phi < self.range_dir[1]):
            return False
        return True
    
    def draw(self, ax, elems, cam_pose):
        for lm in self.lastdata:
            x, y, theta = cam_pose
            dis, dir = lm[0][0], lm[0][1]
            lx = x + dis*math.cos(theta+dir)
            ly = y + dis*math.sin(theta+dir)
            elems += ax.plot([x, lx], [y, ly], color="pink")
    
    @classmethod
    def observe(self, cam_pose, obj_pos):
        diff = obj_pos - cam_pose[0:2]
        dis = np.linalg.norm(diff)
        phi = math.atan2(diff[1], diff[0]) - cam_pose[2]
        while phi >= np.pi:
            phi -= 2*np.pi
        while phi <= -np.pi:
            phi += 2*np.pi
        return np.array([dis, phi]).T


# In[62]:

if __name__ == '__main__':
    w = World(10, 0.1)
    m = Map()
    cam = IdealCamera(m)
    m.append_landmark(LandMark(2, -2))
    m.append_landmark(LandMark(-1, -3))
    m.append_landmark(LandMark(3, 3))
    lin = Agent(0.2, 0.0)
    ver = Agent(0.2, 10.0/180.0*math.pi)
    robot1 = IdealRobot(np.array([2,3,math.pi/6]).T, lin, IdealCamera(m), 'green')
    robot2 = IdealRobot(np.array([-2,-1,math.pi/5*6]).T, ver, IdealCamera(m), 'red')
    robot3 = IdealRobot(np.array([0,0,0]).T, color='blue')
    w.append(robot1)
    w.append(robot2)
    w.append(robot3)
    w.append(m)
    w.draw()