import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation

class PSO:
    def __init__(self, func, d, min, max, particles, w, c1, c2, iterations, topology="gbest"):
        # Історія
        self.best_history = []
        self.swarm_history = []

        # Параметри
        self.func = func
        self.d = d
        self.min = min
        self.max = max
        self.particles = particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.iterations = iterations
        self.topology = topology

        if self.topology == "grid":
            side = int(np.sqrt(self.particles))
            if side * side != self.particles:
                raise ValueError("Кількість частинок має бути квадратом")
            self.grid_side = side
    
    # Сусідство для ring
    def ring_best_indices(self, pbest_val):
        n = self.particles
        k = 2
        best_indices = np.empty(n, dtype=int)
        for i in range(n):
            idxs = [(i + j) % n for j in range(-k, k + 1)]
            best = min(idxs, key=lambda j: pbest_val[j])
            best_indices[i] = best
        return best_indices

    # Сусідство для grid
    def grid_best_indices(self, pbest_val):
        side = self.grid_side
        best_indices = np.empty(self.particles, dtype=int)
        for i in range(self.particles):
            r = i // side
            c = i % side
            neighbors = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < side and 0 <= cc < side:
                        neighbors.append(rr * side + cc)
            best = min(neighbors, key=lambda j: pbest_val[j])
            best_indices[i] = best
        return best_indices

    # Основна функція
    def solve(self):
        X = np.random.uniform(self.min, self.max, (self.particles, self.d))
        V = np.random.uniform(-(self.max - self.min), (self.max - self.min), (self.particles, self.d)) * 0.1

        # Початкова найкраща особиста позиція
        pbest = X.copy()
        pbest_val = np.array([self.func(x) for x in pbest])

        # Початкова найкраща глобальна позиція
        gbest_index = np.argmin(pbest_val)
        gbest = pbest[gbest_index].copy()
        gbest_val = pbest_val[gbest_index]

        for it in range(self.iterations):
            # Знаходження сусідства
            if self.topology == "gbest":
                nbest = np.tile(gbest, (self.particles, 1))

            elif self.topology == "ring":
                best_idxs = self.ring_best_indices(pbest_val)
                nbest = pbest[best_idxs]

            elif self.topology == "grid":
                best_idxs = self.grid_best_indices(pbest_val)
                nbest = pbest[best_idxs]

            else:
                raise ValueError(f"Невідома топологія: {self.topology}")

            r1 = np.random.rand(self.particles, self.d)
            r2 = np.random.rand(self.particles, self.d)

            # Оновлення швидкостей
            V = (self.w * V + self.c1 * r1 * (pbest - X) + self.c2 * r2 * (nbest - X))

            # Оновлення позицій x
            X = X + V

            for i in range(self.d):
                mask_low = X[:, i] < self.min
                mask_high = X[:, i] > self.max

                if np.any(mask_low):
                    X[mask_low, i] = 2 * self.min - X[mask_low, i]
                if np.any(mask_high):
                    X[mask_high, i] = 2 * self.max - X[mask_high, i]

                V[mask_low | mask_high, i] *= -1

            self.swarm_history.append(X.copy())

            # Розрахунок f
            values = np.array([self.func(x) for x in X])

            # Оновлення особистого найкращого
            improved = values < pbest_val
            pbest[improved] = X[improved]
            pbest_val[improved] = values[improved]

            # Оновдення глобального найкращого
            iter_best_val = pbest_val.min()
            if iter_best_val < gbest_val:
                gbest_val = iter_best_val
                gbest = pbest[pbest_val.argmin()].copy()

            self.best_history.append(gbest_val)

        return gbest, gbest_val
    
    # Вивід у ітерацій консоль
    def print_iterations(self):
        if not self.best_history:
            print("History is empty. Run solve() first.")
            return

        total = len(self.best_history)
        for i, val in enumerate(self.best_history):
            print(f"Iteration {i + 1}/{total}, minimum f(x) = {val}")

    # Графік оптимізації f
    def plot_convergence(self):
        plt.figure()
        plt.plot(self.best_history)
        plt.yscale("log")
        plt.xlabel("Iteration")
        plt.ylabel("Minimum f(x) (log scale)")
        plt.title("PSO convergence (log scale)")
        plt.grid(True, which="both", linestyle="--")
        plt.show()

    # Анімація оптимізації
    def animate_optimization(self):
        if not self.swarm_history:
            print("History is empty. Run solve() first")
            return

        swarm = np.array(self.swarm_history)
        x_vals = swarm[:, :, 0]
        y_vals = swarm[:, :, 1]
        frames = len(x_vals)

        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.28)

        grid = 100
        x = np.linspace(self.min, self.max, grid)
        y = np.linspace(self.min, self.max, grid)
        Xg, Yg = np.meshgrid(x, y)

        Z = np.zeros_like(Xg)
        for i in range(grid):
            for j in range(grid):
                p = np.zeros(self.d)
                p[0], p[1] = Xg[i, j], Yg[i, j]
                Z[i, j] = self.func(p)

        ax.contourf(Xg, Yg, Z, levels=40)
        scatter = ax.scatter(x_vals[0], y_vals[0], c='red', s=25)
        # Лінії руху для кожної частинки
        lines = [
            ax.plot([x_vals[0, i], x_vals[0, i]],
                    [y_vals[0, i], y_vals[0, i]],
                    color="blue", linewidth=0.7)[0]
            for i in range(self.particles)
        ]

        ax.set_xlim(self.min, self.max)
        ax.set_ylim(self.min, self.max)
        ax.set_title("PSO Optimization")

        frame_index = [0]
        playing = [False]

        # Оновлення кадру
        def draw_frame(i, from_slider=False):
            frame_index[0] = i
            # Оновлюємо точки
            scatter.set_offsets(np.c_[x_vals[i], y_vals[i]])
            # Оновлюємо лінії (траєкторії)
            if i > 0:
                for p in range(self.particles):
                    lines[p].set_data(
                        [x_vals[i - 1, p], x_vals[i, p]],
                        [y_vals[i - 1, p], y_vals[i, p]]
                    )
            else:
                for p in range(self.particles):
                    lines[p].set_data(
                        [x_vals[0, p], x_vals[0, p]],
                        [y_vals[0, p], y_vals[0, p]]
                    )
            ax.set_title(f"Iteration {i + 1}/{frames}")
            if not from_slider:
                slider.set_val(i)

            fig.canvas.draw_idle()

        # Запуск чи пауза
        def toggle(event):
            playing[0] = not playing[0]
            btn_play.label.set_text("Pause" if playing[0] else "Play")

        # Далі
        def next_frame(event):
            i = min(frame_index[0] + 1, frames - 1)
            draw_frame(i)

        # Назад
        def prev_frame(event):
            i = max(frame_index[0] - 1, 0)
            draw_frame(i)

        # Слайдер
        ax_slider = plt.axes([0.2, 0.16, 0.6, 0.03])
        slider = Slider(ax_slider, "Iteration", 0, frames - 1,
                        valinit=0, valstep=1)

        def slider_update(val):
            draw_frame(int(val), from_slider=True)

        slider.on_changed(slider_update)

        # Кнопки
        ax_play = plt.axes([0.45, 0.05, 0.1, 0.06])
        ax_next = plt.axes([0.58, 0.05, 0.1, 0.06])
        ax_prev = plt.axes([0.32, 0.05, 0.1, 0.06])

        btn_play = Button(ax_play, "Play")
        btn_next = Button(ax_next, "Next")
        btn_prev = Button(ax_prev, "Prev")

        btn_play.on_clicked(toggle)
        btn_next.on_clicked(next_frame)
        btn_prev.on_clicked(prev_frame)

        def update_auto(frame):
            if playing[0]:
                i = (frame_index[0] + 1) % frames
                draw_frame(i)

        ani = FuncAnimation(fig, update_auto, interval=120, cache_frame_data=False)

        plt.show()