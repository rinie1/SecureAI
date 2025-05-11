import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw
import math


class SmoothDigitDrawer:
    def __init__(self, root):
        self.root = root
        self.root.title("Рисование цифры (сглаженное)")

        # Параметры
        self.width, self.height = 280, 280
        self.pen_color = "white"
        self.pen_size = 15
        self.smoothing = True  # Включить сглаживание

        # Холст
        self.canvas = tk.Canvas(root, width=self.width, height=self.height, bg="black")
        self.canvas.pack()

        # Привязка событий
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        # Кнопки
        tk.Button(root, text="Сохранить", command=self.save_image).pack()
        tk.Button(root, text="Очистить", command=self.clear_canvas).pack()

        # Данные для рисования
        self.last_x, self.last_y = None, None
        self.image = Image.new("L", (self.width, self.height), 0)
        self.draw_img = ImageDraw.Draw(self.image)

    def draw(self, event):
        if self.last_x and self.last_y:
            if self.smoothing:
                # Добавляем промежуточные точки для сглаживания
                points = self._get_interpolated_points(self.last_x, self.last_y, event.x, event.y)
                for x, y in points:
                    self._draw_point(x, y)
            else:
                self._draw_point(event.x, event.y)
        self.last_x, self.last_y = event.x, event.y

    def _draw_point(self, x, y):
        """Рисует точку с текущими параметрами"""
        r = self.pen_size // 2
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=self.pen_color, outline="")
        self.draw_img.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def _get_interpolated_points(self, x1, y1, x2, y2, steps=10):
        """Возвращает промежуточные точки между (x1,y1) и (x2,y2)"""
        points = []
        for i in range(steps + 1):
            t = i / steps
            x = x1 + (x2 - x1) * t
            y = y1 + (y2 - y1) * t
            points.append((x, y))
        return points

    def reset(self, event):
        self.last_x = self.last_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.width, self.height), 0)
        self.draw_img = ImageDraw.Draw(self.image)

    def save_image(self):
        self.image.save("digit.jpg", "JPEG")
        messagebox.showinfo("Сохранено", "Изображение сохранено как digit.jpg")


if __name__ == "__main__":
    root = tk.Tk()
    app = SmoothDigitDrawer(root)
    root.mainloop()