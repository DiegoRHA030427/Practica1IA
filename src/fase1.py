"""
Fase 1: búsqueda a ciegas, es decir, sin usar coordenadas ni pistas del mapa.

Todas las búsquedas de este archivo reciben:
    vecinos = {nodo: [(vecino, metros), ...]}   (se arma en mapa.py)
    origen, destino = identificadores de nodo

Y todas regresan un diccionario con el mismo formato, para que las tablas
comparativas salgan parejas:
    "camino":          lista de nodos del origen al destino, o None si no hay forma de llegar
    "metros":          longitud total del camino, en metros
    "saltos":          número de calles recorridas (len(camino) - 1)
    "expandidos":      cuántos nodos sacamos de la frontera para revisarlos
    "frontera_max":    el tamaño más grande que llegó a tener la frontera
    "tiempo_ms":       tiempo de ejecución, en milisegundos
    "orden_expansion": los nodos en el orden en que se expandieron (lo usamos para los snapshots)
"""

import time
import heapq
from collections import deque


def reconstruir_camino(padres, destino):
    """Reconstruye el camino recorriendo el diccionario de padres, del destino hacia el origen."""
    camino = [destino]
    while padres[camino[-1]] is not None:   # el origen es el único nodo cuyo padre es None
        camino.append(padres[camino[-1]])
    camino.reverse()
    return camino


def metros_del_camino(vecinos, camino):
    """Suma la longitud de cada una de las calles que forman el camino."""
    total = 0.0
    for i in range(len(camino) - 1):
        u, v = camino[i], camino[i + 1]
        for vecino, metros in vecinos[u]:
            if vecino == v:
                total += metros
                break
    return total


def armar_resultado(vecinos, padres, destino, encontrado, expandidos,
                    frontera_max, inicio, orden_expansion):
    """Arma el diccionario de salida que comparten todas las búsquedas."""
    tiempo_ms = (time.perf_counter() - inicio) * 1000
    if encontrado:
        camino = reconstruir_camino(padres, destino)
        metros = metros_del_camino(vecinos, camino)
        saltos = len(camino) - 1
    else:
        camino, metros, saltos = None, 0.0, 0
    return {
        "camino": camino,
        "metros": metros,
        "saltos": saltos,
        "expandidos": expandidos,
        "frontera_max": frontera_max,
        "tiempo_ms": tiempo_ms,
        "orden_expansion": orden_expansion,
    }


def bfs(vecinos, origen, destino):
    """Búsqueda en anchura: encuentra el camino con menos saltos, sin importar los metros.

    Trabaja con una cola FIFO, así que primero revisa todos los nodos que están
    a un salto, luego los que están a dos, y así sucesivamente. Por eso el primer
    camino que llega al destino es siempre el de menos calles.
    """
    inicio = time.perf_counter()

    # Usamos deque porque saca elementos del frente en O(1); con una lista
    # normal, pop(0) tendría que recorrer toda la lista cada vez.
    frontera = deque([origen])

    # padres[n] nos dice desde qué nodo llegamos a n. Además nos sirve como
    # registro de visitados: si un nodo ya está aquí, no lo volvemos a meter.
    padres = {origen: None}

    expandidos = 0
    frontera_max = 1
    orden_expansion = []
    encontrado = False

    while frontera:
        actual = frontera.popleft()
        expandidos += 1
        orden_expansion.append(actual)

        # Revisamos si llegamos a la meta al sacar el nodo (igual que en UCS),
        # para que el conteo de expandidos sea comparable entre los tres algoritmos.
        if actual == destino:
            encontrado = True
            break

        for vecino, _metros in vecinos[actual]:   # a BFS no le importan los metros
            if vecino not in padres:
                padres[vecino] = actual
                frontera.append(vecino)

        frontera_max = max(frontera_max, len(frontera))

    return armar_resultado(vecinos, padres, destino, encontrado, expandidos,
                           frontera_max, inicio, orden_expansion)


def dfs(vecinos, origen, destino):
    """Búsqueda en profundidad: encuentra algún camino, avanzando lo más lejos posible antes de regresar.

    Usa una pila LIFO (una lista de Python con append y pop), así que el último
    nodo en entrar es el primero que se revisa. No garantiza el camino más corto.

    La hicimos iterativa y no recursiva porque, con miles de nodos, la recursión
    podría rebasar el límite de llamadas de Python y lanzar un RecursionError.
    """
    inicio = time.perf_counter()

    # En la pila guardamos parejas (nodo, nodo del que venimos). Guardamos el padre
    # junto al nodo porque un mismo nodo puede entrar varias veces a la pila desde
    # distintos vecinos, y el padre que cuenta es el de la vez en que se expande.
    pila = [(origen, None)]

    # Aquí van los nodos que ya expandimos. En un mapa de calles esto es indispensable:
    # hay ciclos (como dar la vuelta a la manzana) y sin este registro DFS nunca terminaría.
    visitados = set()
    padres = {}

    expandidos = 0
    frontera_max = 1
    orden_expansion = []
    encontrado = False

    while pila:
        actual, padre = pila.pop()

        if actual in visitados:
            continue
        visitados.add(actual)
        padres[actual] = padre
        expandidos += 1
        orden_expansion.append(actual)

        if actual == destino:
            encontrado = True
            break

        for vecino, _metros in vecinos[actual]:   # DFS tampoco toma en cuenta los metros
            if vecino not in visitados:
                pila.append((vecino, actual))

        frontera_max = max(frontera_max, len(pila))

    return armar_resultado(vecinos, padres, destino, encontrado, expandidos,
                           frontera_max, inicio, orden_expansion)


def ucs(vecinos, origen, destino):
    """Búsqueda de costo uniforme: encuentra el camino más corto en metros.

    Usa una cola de prioridad (heapq) y siempre expande el nodo con el menor costo
    acumulado g(n) desde el origen. Cuando el destino sale de la cola ya no puede
    haber un camino más barato, porque todo lo que queda en la cola cuesta lo
    mismo o más, y ninguna calle mide menos de cero.
    """
    inicio = time.perf_counter()

    # heapq trabaja sobre una lista normal y siempre deja al frente el elemento menor.
    # Guardamos tuplas (costo, nodo) porque Python compara primero el costo.
    frontera = [(0.0, origen)]

    # mejor_costo[n] es el costo más bajo con el que hemos llegado a n hasta el momento.
    mejor_costo = {origen: 0.0}
    padres = {origen: None}
    visitados = set()   # nodos ya expandidos; su costo ya es definitivo

    expandidos = 0
    frontera_max = 1
    orden_expansion = []
    encontrado = False

    while frontera:
        costo, actual = heapq.heappop(frontera)

        # Un mismo nodo puede estar varias veces en la cola con costos distintos,
        # si más adelante encontramos un camino más barato. Solo cuenta la primera
        # vez que sale, que es la más barata; las demás copias las ignoramos.
        if actual in visitados:
            continue
        visitados.add(actual)
        expandidos += 1
        orden_expansion.append(actual)

        if actual == destino:
            encontrado = True
            break

        for vecino, metros in vecinos[actual]:
            nuevo_costo = costo + metros
            # Solo lo agregamos si es la primera vez que lo vemos o si encontramos
            # un camino más barato que el que ya teníamos.
            if vecino not in mejor_costo or nuevo_costo < mejor_costo[vecino]:
                mejor_costo[vecino] = nuevo_costo
                padres[vecino] = actual
                heapq.heappush(frontera, (nuevo_costo, vecino))

        frontera_max = max(frontera_max, len(frontera))

    return armar_resultado(vecinos, padres, destino, encontrado, expandidos,
                           frontera_max, inicio, orden_expansion)


def nodos_alcanzables(vecinos, origen):
    """Regresa el conjunto de todos los nodos a los que se puede llegar desde el origen.

    Es un BFS sin destino: sigue avanzando hasta que la cola se vacía, o sea,
    hasta agotar todo lo que se puede alcanzar. Como el grafo es dirigido (hay
    calles de un solo sentido), que A llegue a B no quiere decir que B llegue a A.
    """
    alcanzados = {origen}
    cola = deque([origen])
    while cola:
        actual = cola.popleft()
        for vecino, _metros in vecinos[actual]:
            if vecino not in alcanzados:
                alcanzados.add(vecino)
                cola.append(vecino)
    return alcanzados


def verificar_entregas(vecinos, deposito, destinos):
    """Indica, para cada destino, si se puede llegar a él desde el depósito: {destino: True/False}.

    Calculamos los nodos alcanzables una sola vez y después cada consulta es
    solo buscar en un set, lo cual es inmediato. Así evitamos correr un BFS por destino.
    """
    alcanzables = nodos_alcanzables(vecinos, deposito)
    return {destino: (destino in alcanzables) for destino in destinos}


if __name__ == "__main__":
    from mapa import cargar_grafo, pares_origen_destino, nodo_mas_cercano, DEPOSITO

    vecinos, coords = cargar_grafo()
    pares = pares_origen_destino(coords)

    algoritmos = [("BFS", bfs), ("DFS", dfs), ("UCS", ucs)]

    print(f"{'Alg':4s} {'Destino':28s} {'Metros':>8s} {'Saltos':>7s} {'Expand.':>8s} {'Front.max':>9s} {'ms':>7s}")
    for nombre, origen, destino in pares:
        for nombre_alg, funcion in algoritmos:
            r = funcion(vecinos, origen, destino)
            if r["camino"] is None:
                print(f"{nombre_alg:4s} {nombre:28s} INALCANZABLE (expandidos: {r['expandidos']})")
            else:
                print(f"{nombre_alg:4s} {nombre:28s} {r['metros']:8.0f} {r['saltos']:7d} {r['expandidos']:8d} "
                      f"{r['frontera_max']:9d} {r['tiempo_ms']:7.2f}")
        print()

    deposito, _ = nodo_mas_cercano(coords, DEPOSITO[0], DEPOSITO[1])
    alcanzables = nodos_alcanzables(vecinos, deposito)
    no_alcanzables = [n for n in vecinos if n not in alcanzables]
    print("Accesibilidad desde el depósito:")
    print("  Alcanzables:", len(alcanzables), "de", len(vecinos), "nodos")
    print("  NO alcanzables:", len(no_alcanzables))

    # Los lugares que elegimos como destinos deberían ser todos alcanzables.
    destinos = [destino for _nombre, _origen, destino in pares]
    resultado = verificar_entregas(vecinos, deposito, destinos)
    for nombre, _origen, destino in pares:
        print(f"  {nombre:28s} alcanzable: {resultado[destino]}")

    # Probamos también con un nodo inalcanzable: UCS debería regresar el camino como None.
    if no_alcanzables:
        ejemplo = no_alcanzables[0]
        r = ucs(vecinos, deposito, ejemplo)
        print(f"\n  Ejemplo inalcanzable: nodo {ejemplo} en {coords[ejemplo]}")
        print(f"  UCS -> camino: {r['camino']}, expandidos: {r['expandidos']}")
