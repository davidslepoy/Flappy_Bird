from agentes.base import Agent
import numpy as np
import tensorflow as tf

class NNAgent(Agent):
    """
    Agente que utiliza una red neuronal entrenada para aproximar la Q-table.
    La red debe estar guardada como TensorFlow SavedModel.
    """
    def __init__(self, actions, game=None, model_path='flappy_q_nn_model.keras'):
        super().__init__(actions, game)
        # Cargar el modelo entrenado
        self.model = tf.keras.models.load_model(model_path)
        self.actions = actions

            
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
                bin_value(state['next_pipe_bottom_y'], bins['pipe_bottom_y'], 0, 512)
            )
    
    def act(self, state):
        """
        Transforma el estado en un array de entrada para la red y devuelve la acción con mayor Q-value.
        """
        discrete_state = self.discretize_state(state)
            
        # Convertir a array y normalizar como durante el entrenamiento
        input_state = np.array(discrete_state, dtype=np.float32)
        input_state = input_state.reshape(1, -1)  # Reshape a (1, 5)
        
        q_values = self.model.predict(input_state, verbose=0)[0]  # [0] para quedarte con el vector plano
        return self.actions[int(np.argmax(q_values))]

