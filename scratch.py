import tkinter as tk
import math
import random

class SphereTest:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("250x250")
        self.root.config(bg="#010101")
        self.canvas = tk.Canvas(self.root, width=250, height=250, bg="#010101", highlightthickness=0)
        self.canvas.pack()
        
        self.num_points = 180
        self.points = []
        phi = math.pi * (3. - math.sqrt(5.))
        for i in range(self.num_points):
            y = 1 - (i / float(self.num_points - 1)) * 2
            radius = math.sqrt(1 - y * y)
            theta = phi * i
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            is_red = random.random() > 0.4
            color = "#FF2222" if is_red else "#E0E0E0"
            self.points.append([x, y, z, color])
            
        self.connections = []
        for i, p1 in enumerate(self.points):
            for j, p2 in enumerate(self.points):
                if i < j:
                    dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
                    if dist < 0.45:
                        self.connections.append((i, j))
                        
        self.angle_x = 0
        self.angle_y = 0
        self.animate()
        
    def animate(self):
        self.canvas.delete("all")
        self.angle_x += 0.03
        self.angle_y += 0.04
        
        cx, cy = 125, 125
        scale = 90
        
        projected = []
        for p in self.points:
            x, y, z, color = p
            xy = y * math.cos(self.angle_x) - z * math.sin(self.angle_x)
            xz = y * math.sin(self.angle_x) + z * math.cos(self.angle_x)
            yx = x * math.cos(self.angle_y) + xz * math.sin(self.angle_y)
            yz = -x * math.sin(self.angle_y) + xz * math.cos(self.angle_y)
            projected.append((cx + yx * scale, cy + xy * scale, yz, color))
            
        for i, j in self.connections:
            p1, p2 = projected[i], projected[j]
            if p1[2] > -0.2 and p2[2] > -0.2:
                # Color the line based on the points
                lcolor = "#772222" if p1[3] == "#FF2222" and p2[3] == "#FF2222" else "#666666"
                self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill=lcolor, width=1)
                
        for p in projected:
            if p[2] > -0.5:
                sz = max(1.0, (p[2] + 1.5) * 1.5)
                self.canvas.create_oval(p[0]-sz, p[1]-sz, p[0]+sz, p[1]+sz, fill=p[3], outline="")
                
        self.root.after(30, self.animate)

if __name__ == "__main__":
    app = SphereTest()
    app.root.after(3000, app.root.destroy)
    app.root.mainloop()
    print("Success")
