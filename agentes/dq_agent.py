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
        """
        bins = {
            'player_y': 20,
            'player_vel': 10,
            'pipe_dist': 15,
            'pipe_top_y': 20,
            'pipe_bottom_y': 20,
        }
        def bin_value(value, n_bins, min_val, max_val):
            bin_size = (max_val - min_val) / n_bins
            return int(min(max((value - min_val) // bin_size, 0), n_bins - 1))

        return (
            bin_value(state['player_y'], bins['player_y'], 0, 512),
            bin_value(state['player_vel'], bins['player_vel'], -20, 20),
            bin_value(state['next_pipe_dist_to_player'], bins['pipe_dist'], 0, 288),
            bin_value(state['next_pipe_top_y'], bins['pipe_top_y'], 0, 512),
            bin_value(state['next_pipe_bottom_y'], bins['pipe_bottom_y'], 0, 512),
            # bin_value(state['next_next_pipe_dist_to_player'], bins['pipe_dist'], 0, 288),
            # bin_value(state['next_next_pipe_top_y'], bins['pipe_top_y'], 0, 512),
            # bin_value(state['next_next_pipe_bottom_y'], bins['pipe_bottom_y'], 0, 512),
        )


    def act(self, state):
        """
        Elige una acción usando epsilon-greedy sobre la Q-table.
        """
        discrete_state = self.discretize_state(state)
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_values = self.q_table[discrete_state]
        return self.actions[int(np.argmax(q_values))]


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
            player_y = state["player_y"]
            pipe_top = state["next_pipe_top_y"]
            pipe_bottom = state["next_pipe_bottom_y"]
            gap_center = (pipe_top + pipe_bottom) / 2
            dist_to_center = abs(gap_center - player_y)
            norm_distance = dist_to_center / 512
            reward -= norm_distance *0.2
            
        new_q = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)
        self.q_table[discrete_state][action_idx] = new_q
        # BONUS: Recompensa baja por estar cerca del centro del hueco (solo si no está muerto)
    


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
