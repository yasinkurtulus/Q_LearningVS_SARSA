"""MATLAB örneğindeki 6 × 6 ortamda SARSA ve Q-learning."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


SIZE = 6
START = (0, 0)
GOAL = (5, 5)
# MATLAB koordinatları Python için bir azaltıldı.
OBSTACLES = {(1, 1), (1, 2), (2, 2), (3, 3), (4, 4),
             (2, 5), (5, 2), (1, 4), (3, 0)}
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ARROWS = ["↑", "↓", "←", "→"]
ALPHA, GAMMA, EPSILON = 0.1, 0.9, 0.1
EPISODES = 5000


def index(state):
    row, col = state
    return row * SIZE + col


def step(state, action):
    row, col = state
    dr, dc = ACTIONS[action]
    next_state = (row + dr, col + dc)
    if not all(0 <= coordinate < SIZE for coordinate in next_state):
        return state, -10
    if next_state in OBSTACLES:
        return next_state, -10  # MATLAB örneğinde ajan bu kareye giriyor.
    if next_state == GOAL:
        return next_state, 10
    return next_state, -1


def choose_action(values, rng):
    """Eğitim politikası: ε-greedy, eşit değerlerde rastgele seçim."""
    if rng.random() < EPSILON:
        return rng.integers(len(ACTIONS))
    best_actions = np.flatnonzero(values == values.max())
    return rng.choice(best_actions)


def train(method, rng):
    q = np.zeros((SIZE * SIZE, len(ACTIONS)))
    history = []
    for episode in range(1, EPISODES + 1):
        state = START
        action = choose_action(q[index(state)], rng)
        while state != GOAL:
            next_state, reward = step(state, action)
            current = index(state)
            following = index(next_state)
            if next_state == GOAL:
                target = reward
            elif method == "SARSA":
                next_action = choose_action(q[following], rng)
                target = reward + GAMMA * q[following, next_action]
            else:  # Q-learning
                target = reward + GAMMA * q[following].max()

            q[current, action] += ALPHA * (target - q[current, action])
            state = next_state
            if state != GOAL:
                if method == "SARSA":
                    action = next_action
                else:
                    action = choose_action(q[following], rng)

        if episode == 1 or episode % 50 == 0 or episode == EPISODES:
            history.append((episode, q.copy()))
    return q, history


def print_policy(q):
    for row in range(SIZE):
        cells = []
        for col in range(SIZE):
            state = (row, col)
            if state == GOAL:
                cells.append("GG")
            elif state in OBSTACLES:
                cells.append("XX")
            else:
                cells.append(ARROWS[q[index(state)].argmax()])
        print(" ".join(f"{cell:>2}" for cell in cells))


def animate(results):
    fig, axes = plt.subplots(1, 2, figsize=(9, 6), constrained_layout=True)
    # Ortak renk ölçeği, iki tablodaki değerlerin karşılaştırılmasını sağlar.
    low = min(q.min() for history in results.values() for _, q in history)
    high = max(q.max() for history in results.values() for _, q in history)
    images = []
    for ax, history in zip(axes, results.values()):
        image = ax.imshow(history[0][1], aspect="auto", cmap="gray_r",
                          vmin=low, vmax=high)
        ax.set_xticks(range(4), ARROWS)
        ax.set_xlabel("Eylem")
        ax.set_ylabel("Durum indeksi")
        images.append(image)
    fig.colorbar(images[0], ax=list(axes), label="Q")

    def draw(frame):
        for ax, image, (name, history) in zip(axes, images, results.items()):
            episode, q = history[frame]
            image.set_data(q)
            ax.set_title(f"{name} | Bölüm {episode}")

    animation = FuncAnimation(fig, draw, frames=len(results["SARSA"]),
                              interval=50, repeat=False)
    plt.show()
    return animation


if __name__ == "__main__":
    histories = {}
    for name in ("SARSA", "Q-learning"):
        q_table, histories[name] = train(name, np.random.default_rng(42))
        print(f"\n{name} Q-tablosu (yukarı, aşağı, sol, sağ)")
        print(q_table.round(2))
        print(f"\n{name} greedy politikası")
        print_policy(q_table)
    animate(histories)
