import numpy as np
import random


# PARAMETROS
alpha = 0.2
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9
episodes = 3000

# Estado carta ( 1 al 20) para mi esto es un numero
# Acciones: 0 = menor, 1 = mayor
actions = [0, 1]
num_actions = len(actions)

Q = {}


def get_q_values(state): #funcion aux
    
    if state not in Q:
        Q[state] = np.zeros(num_actions)
    return Q[state]


def step(current_card, action): # se saca una carta del 1 al 20
    
    while True:
        next_card = random.randint(1, 20)
        if next_card != current_card:
            break
            
    game_over = False
    

    if action == 1: # El agente apostó a que la siguiente es MAYOR
        if next_card > current_card:
            reward = 30   #verdadero
        else:
            reward = -30  #falso
            
    elif action == 0: # El agente verifica si la siguiente carta es menor
        if next_card < current_card:
            reward = 30   #verdadero
        else:
            reward = -30  #falso

    
    game_over = True  #termina de verificar
    
    return next_card, reward, game_over


# ENTRENAMIENTO

print("Adivinar el numero (Mayor/Menor)")

for episode in range(episodes):
   
    current_card = random.randint(1, 20)
    game_over = False
    
    while not game_over:
        q_values = get_q_values(current_card)
        
        # Epsilon-greedy
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)
        else:
            action = np.argmax(q_values)
            
        next_card, reward, game_over = step(current_card, action)
        
        # En Q-Learning, si el juego termina, el valor futuro es 0
        old_value = q_values[action]
        q_values[action] = old_value + alpha * (reward - old_value)
        
        current_card = next_card
        
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

print(f"juego de cartas: {len(Q)}")


print("DEMOSTRACIÓN") # adivinar el numero


# Hacemos 3 pruebas con cartas críticas para ver qué decide la Tabla Q
cartas_prueba = [2, 19, 11]
nombres_acciones = {0: "MENOR ↓", 1: "MAYOR ↑"}

for i, carta in enumerate(cartas_prueba):
    q_values = get_q_values(carta)
    action = np.argmax(q_values) 
    
    print(f"Prueba {i+1} | Carta en mano: [{carta}]")
    print(f" -> Valores Q para esta carta: [Menor: {q_values[0]:.1f}, Mayor: {q_values[1]:.1f}]")
    print(f" -> La Tabla Q decide apostar de forma óptima a: {nombres_acciones[action]}")
  