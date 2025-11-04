#Nombre: Gomez Garcia Flores Dafne July
#Carrera: Ingenieria de Sistemas

# Crear un agente que resuelva: B(0,1,2,3,4,5)
#El estado seria la lista actual seria la desordenada lo cual a que convertir de manera ordenada
#La accion intercambiar el elemento i + 1
#La recompesa ordenar +1
# Configuracion

import random
lista_objetivo = [0, 1, 2, 3, 4, 5]     # Lista ordenada
lista_inicial = [4, 5, 3, 1, 2, 0]      # Lista desordenada
max_steps = 20                          # Límite de pasos por episodio

alpha = 0.1    # Tasa de aprendizaje ( que tanto se ajusta)
gamma = 0.9    # Descuento de recompensa futura
epsilon = 0.2  # Probabilidad de exploración aleatoria
episodes = 4000    #Numero de episodios de entrenamientos 

# estado (tupla de lista) acción (índice i) donde se guarda los valores
q_table = {}

# Funciones
def get_state(lista):
    return tuple(lista)  # convertir a tupla para usar como llave en Q-Table

def get_valid_actions(lista):
    # Puede mover cualquier posición excepto la última
    return list(range(len(lista)-1))  # aqui el numero 5 

def swap(lista, i):
    nueva_lista = lista.copy()
    nueva_lista[i], nueva_lista[i+1] = nueva_lista[i+1], nueva_lista[i]
    return nueva_lista
# elegir accion
def choose_action(state, valid_actions):
    if state not in q_table:
        q_table[state] = {a:0 for a in valid_actions}
    if random.uniform(0,1) < epsilon:
        return random.choice(valid_actions)
    q_vals = q_table[state]
    return max(valid_actions, key=lambda a: q_vals.get(a,0))

def compute_reward(lista):
    if lista == lista_objetivo:
        return 1  
    # Lista ordenada
    # Penalización por estar desordenada (cuantos pares fuera de orden)
    penalty = sum(1 for i in range(len(lista)-1) if lista[i] > lista[i+1])
    return -penalty / len(lista)

def update_q(state, action, reward, next_state):
    if state not in q_table:
        q_table[state] = {action:0}
    if action not in q_table[state]:
        q_table[state][action] = 0
    next_max = 0
    if next_state in q_table and q_table[next_state]:
        next_max = max(q_table[next_state].values())
    q_table[state][action] += alpha * (reward + gamma*next_max - q_table[state][action])

# Entrenamiento 
for ep in range(episodes):
    lista = lista_inicial.copy()
    steps = 0
    done = False
    while not done and steps < max_steps:
        state = get_state(lista)
        valid_actions = get_valid_actions(lista)
        action = choose_action(state, valid_actions)
        nueva_lista = swap(lista, action)
        next_state = get_state(nueva_lista)
        reward = compute_reward(nueva_lista)
        update_q(state, action, reward, next_state)
        lista = nueva_lista
        steps += 1
        if reward == 1:
            done = True

print("Entrenamiento terminado ✅")


# Simulación de lo que interpreta 
lista = lista_inicial.copy()
steps = 0
print(f"Lista inicial: {lista}")
while lista != lista_objetivo and steps < max_steps:
    state = get_state(lista)
    valid_actions = get_valid_actions(lista)
    action = choose_action(state, valid_actions)
    lista = swap(lista, action)
    steps += 1
    print(f"Paso {steps}: {lista}")

if lista == lista_objetivo:
    print("Lista ordenada con éxito")
else:
    print("No se logró ordenar la lista en los pasos permitidos")
