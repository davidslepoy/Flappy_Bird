from agentes.base import Agent
import numpy as np
from collections import defaultdict
import pickle
import random

class QAgent(Agent):
    """
    Agente de Q-Learning.
    Completar la discretización del estado y la función de acción.
    """
    def __init__(self, actions, game=None, learning_rate=0.1, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, load_q_table_path="flappy_birds_q_table.pkl"):
        super().__init__(actions, game)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        if load_q_table_path:
            try:
                with open(load_q_table_path, 'rb') as f:
                    q_dict = pickle.load(f)
                self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), q_dict)
                print(f"Q-table cargada desde {load_q_table_path}")
            except FileNotFoundError:
                print(f"Archivo Q-table no encontrado en {load_q_table_path}. Se inicia una nueva Q-table vacía.")
                self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        else:
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        # TODO: Definir parámetros de discretización según el entorno

    def discretize_state(self, state):
        """
        Discretiza el estado continuo en un estado discreto (tupla).
        COMPLETAR: Implementar la discretización adecuada para el entorno.
        """

        # Definir dimensiones estándar del juego

        screen_width = self.game.width
        screen_height = self.game.height

        # Bins por variable
        n_bins_y = 10
        n_bins_vel = 5
        n_bins_dist = 10

        # Normalizar posiciones y distancias
        def discretize(value, max_value, bins):
            norm = value / max_value
            return min(int(norm * bins), bins - 1)

        def discretize_vel(value, min_val=-10, max_val=10, bins=5):
            norm = (value - min_val) / (max_val - min_val)
            return min(int(norm * bins), bins - 1)

        # Discretizar cada variable del estado
        player_y_bin = discretize(state["player_y"], screen_height, n_bins_y)
        player_vel_bin = discretize_vel(state["player_vel"])
        
        pipe_dist_bin = discretize(state["next_pipe_dist_to_player"], screen_width, n_bins_dist)
        pipe_top_y_bin = discretize(state["next_pipe_top_y"], screen_height, n_bins_y)
        pipe_bottom_y_bin = discretize(state["next_pipe_bottom_y"], screen_height, n_bins_y)

        next_pipe_dist_bin = discretize(state["next_next_pipe_dist_to_player"], screen_width, n_bins_dist)
        next_pipe_top_y_bin = discretize(state["next_next_pipe_top_y"], screen_height, n_bins_y)
        next_pipe_bottom_y_bin = discretize(state["next_next_pipe_bottom_y"], screen_height, n_bins_y)

        # Retornar el estado discretizado como tupla
        return (
            player_y_bin,
            player_vel_bin,
            pipe_dist_bin,
            pipe_top_y_bin,
            pipe_bottom_y_bin,
            next_pipe_dist_bin,
            next_pipe_top_y_bin,
            next_pipe_bottom_y_bin
        )

    def act(self, state):
        """
        Elige una acción usando epsilon-greedy sobre la Q-table.
        COMPLETAR: Implementar la política epsilon-greedy.
        """
        # Discretizar el estado continuo
        discrete_state = self.discretize_state(state)

        # Inicializar la entrada en la Q-table si no existe
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = [0.0 for _ in self.actions]

        # Estrategia epsilon-greedy
        if random.random() < self.epsilon:
            # Explorar: elegir acción aleatoria
            return random.choice(self.actions)
        else:
            # Explotar: elegir la acción con mayor Q-valor
            q_values = self.q_table[discrete_state]
            max_q = max(q_values)
            # Buscar los índices con máximo Q
            best_action_indices = [i for i, q in enumerate(q_values) if q == max_q]
            best_action_index = random.choice(best_action_indices)
            return self.actions[best_action_index] 
        
    def update(self, state, action, reward, next_state, done):
        """
        Actualiza la Q-table usando la regla de Q-learning.
        """
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        action_idx = self.actions.index(action)
        # Inicializar si el estado no está en la Q-table
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(len(self.actions))
        if discrete_next_state not in self.q_table:
            self.q_table[discrete_next_state] = np.zeros(len(self.actions))
        current_q = self.q_table[discrete_state][action_idx]
        max_future_q = 0
        if not done:
            max_future_q = np.max(self.q_table[discrete_next_state])
        new_q = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)
        self.q_table[discrete_state][action_idx] = new_q

    def decay_epsilon(self):
        """
        Disminuye epsilon para reducir la exploración con el tiempo.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_table(self, path):
        """
        Guarda la Q-table en un archivo usando pickle.
        """
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Q-table guardada en {path}")

    def load_q_table(self, path):
        """
        Carga la Q-table desde un archivo usando pickle.
        """
        import pickle
        try:
            with open(path, 'rb') as f:
                q_dict = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), q_dict)
            print(f"Q-table cargada desde {path}")
        except FileNotFoundError:
            print(f"Archivo Q-table no encontrado en {path}. Se inicia una nueva Q-table vacía.")
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
