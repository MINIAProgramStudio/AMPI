# Travaling Salesman Problem
import numpy as np
from tqdm import tqdm
from turtle import Turtle
import turtle
class TSP:
    def __init__(self, vertices, circle = True, init_progressbar = False):
        self.screen = turtle.Screen()
        self.circle = circle
        if circle:
            circle_iterator = range(vertices)
            if init_progressbar:
                circle_iterator = tqdm(circle_iterator, desc = "TSP creating circle with "+str(vertices)+" vertices")
            self.vertices = np.array([
                [np.cos(rad*2*np.pi/vertices), np.sin(rad*2*np.pi/vertices)] for rad in circle_iterator
            ])
        else:
            self.vertices = np.random.rand(vertices,2)

        matrix_iterator = range(vertices)
        if init_progressbar:
            matrix_iterator = tqdm(matrix_iterator, desc = "TSP calculating distances")
        self.matrix = np.array([
            [np.sqrt(np.sum(np.power(self.vertices[i]-self.vertices[j], 2))) for j in range(vertices)] for i in matrix_iterator
        ])
        for _ in range(vertices):
            self.matrix[_][_] = float("inf")

    def reset_screens(self):
        for t in self.screen.turtles():
            t.reset()
            t.hideturtle()

    def check_path(self, vertices_list, cycle = True):
        if len(vertices_list) < 2:
            raise Exception("TSP.check_path ERROR: path must be at least two vertices long")
        length = 0
        for i in range(len(vertices_list)-1):
            length += self.matrix[vertices_list[i]][vertices_list[i+1]]
        if cycle:
            length += self.matrix[vertices_list[-1]][vertices_list[0]]
        return length

    def draw_graph(self, area_size = 700, path = None):
        area_coef = area_size*0.4*(2**(not self.circle))
        turtle.delay(0)
        turtle.screensize(area_size, area_size)

        self.reset_screens()
        self.screen.setup(area_size,area_size)
        self.screen.tracer(False)
        t = Turtle()
        t.hideturtle()
        t.speed(0)

        if len(self.vertices)<=25:
            t.color("gray")
            t.width(1)
            for i in range(len(self.vertices)-1):
                for j in range(i+1, len(self.vertices)):
                    t.penup()

                    t.setpos(self.vertices[i]*area_coef-(not self.circle)*area_size*0.4)
                    t.pendown()

                    t.goto(self.vertices[j]*area_coef-(not self.circle)*area_size*0.4)

        if path:
            t.color("gold")
            t.width(10)
            t.penup()
            t.setpos(self.vertices[path[0]]*area_coef-(not self.circle)*area_size*0.4)
            t.pendown()
            for vertice in path:
                t.goto(self.vertices[vertice]*area_coef-(not self.circle)*area_size*0.4)
            t.goto(self.vertices[path[0]]*area_coef-(not self.circle)*area_size*0.4)


        circle_size = min(area_coef / len(self.vertices),10)
        t.color("red")
        for vertice in self.vertices:
            t.penup()
            t.setpos(vertice*area_coef-np.array([0, circle_size])-(not self.circle)*area_size*0.4)
            t.pendown()
            t.fillcolor("red")
            t.begin_fill()
            t.circle(circle_size,steps=6)
            t.end_fill()

        self.screen.update()