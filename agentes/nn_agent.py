from agentes.base import Agent
import numpy as np
import tensorflow as tf

class NNAgent(Agent):
    """
    Agente que utiliza una red neuronal entrenada para aproximar la Q-table.
    La red debe estar guardada como TensorFlow SavedModel.
    """
    def __init__(self, actions, game=None, model_path='flappy_q_nn_model'):
        super().__init__(actions, game)
        # Cargar el modelo entrenado
        self.model = tf.keras.models.load_model(model_path)

    def act(self, state):
        """
        Transforma el estado en un array de entrada para la red y devuelve la acción con mayor Q-value.
        """
        input_state = np.array([
            state['player_y'],
            state['player_vel'],
            state['next_pipe_dist_to_player'],
            state['next_pipe_top_y'],
            state['next_pipe_bottom_y'],
            # state['next_next_pipe_dist_to_player'],
            # state['next_next_pipe_top_y'],
            # state['next_next_pipe_bottom_y']
        ], dtype=np.float32).reshape(1, -1)

        q_values = self.model.predict(input_state, verbose=0)
        return self.actions[int(np.argmax(q_values))]

